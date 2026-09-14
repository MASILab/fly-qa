"""Event bus broadcasting per-image pipeline results (including real simulated
activity for the dashboard's "neurons firing" animation) to websocket clients.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
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
    asyncio loop running the websocket server.

    A browser cold-start (page load, 3D scene init, websocket handshake) is
    slower than the first few images can be processed, so a subscriber that
    connects late would otherwise miss those early "result" events with no
    way to recover them. A bounded replay buffer fixes that: every new
    subscriber is caught up with the events published so far before it
    starts receiving live ones. `_history` is only ever touched from
    `_dispatch`, and `subscribe` always runs on the same asyncio loop thread
    as `_dispatch` (scheduled via `call_soon_threadsafe`), so there's no race
    between replay and live dispatch.
    """

    def __init__(self, history_limit: int = 500) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: deque[ResultEvent] = deque(maxlen=history_limit)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        for event in self._history:
            queue.put_nowait(event)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: ResultEvent) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._dispatch, event)

    def _dispatch(self, event: ResultEvent) -> None:
        self._history.append(event)
        for queue in list(self._subscribers):
            queue.put_nowait(event)
