"""Local FastAPI dashboard: shows the image being processed, real simulated
"neuron firing" activity for a sampled subset of the actual connectome, and
the pass/fail/flag verdict, live as fly-qa works through a folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fly_qa.events import EventBus
from fly_qa.viz import VizSubset

logger = logging.getLogger("fly_qa.webapp")

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    event_bus: EventBus,
    root: Path | None = None,
    viz_subset: VizSubset | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        event_bus.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    root_resolved = root.resolve() if root is not None else None

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/viz-subset")
    async def viz_subset_endpoint():
        if viz_subset is None:
            return {"nodes": [], "edges": []}
        return viz_subset.to_json_dict()

    @app.get("/api/image")
    async def image(path: str):
        if root_resolved is None:
            raise HTTPException(status_code=404, detail="no root configured")

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root_resolved / candidate
        candidate = candidate.resolve()

        if root_resolved != candidate and root_resolved not in candidate.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        if candidate.suffix.lower() != ".png" or not candidate.is_file():
            raise HTTPException(status_code=404, detail="not found")

        return FileResponse(candidate, media_type="image/png")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        queue = event_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_text(event.to_json())
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(queue)

    return app


def find_free_port(preferred: int = 8420) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free port for the dashboard")


class DashboardServer:
    """Runs the FastAPI app in a background thread so the CLI's main thread can
    run the (synchronous, expensive) connectome simulation loop."""

    def __init__(
        self,
        event_bus: EventBus,
        root: Path | None = None,
        viz_subset: VizSubset | None = None,
        port: int | None = None,
    ):
        self.event_bus = event_bus
        self.port = port or find_free_port()
        self.app = create_app(event_bus, root=root, viz_subset=viz_subset)
        self._config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(self._config)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, open_browser: bool = True) -> None:
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        import time

        for _ in range(50):
            if getattr(self._server, "started", False):
                break
            time.sleep(0.05)
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(self.url)

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
