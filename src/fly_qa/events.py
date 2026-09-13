"""Event bus broadcasting per-image pipeline results (including real simulated
activity for the dashboard's "neurons firing" animation) to websocket clients.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

EventType = Literal["started", "result", "finished"]


@dataclass
class ResultEvent:
    type: EventType = "result"
    path: str = ""
    precheck_passed: bool = True
    precheck_findings: list[str] = field(default_factory=list)
    fed_to_connectome: bool = False
    defect_score: float | None = None
    confidence_signal: float | None = None
    verdict: str = ""
    calibrated: bool = False
    total_images: int = 0
    processed_images: int = 0
    tally: dict = field(default_factory=lambda: {"pass": 0, "fail": 0, "flag": 0})
    # step_activity[step][i] = rate of viz-subset node i at that integration step,
    # in the same node order as the /api/viz-subset response. None if not fed to
    # the connectome (precheck failed).
    step_activity: list[list[float]] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EventBus:
    """Bridges the (synchronous, possibly-another-thread) processing loop to the
    asyncio loop running the websocket server."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: ResultEvent) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: ResultEvent) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)
