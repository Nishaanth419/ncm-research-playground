"""In-memory fan-out of generation events from the worker thread to WebSockets.

Every event published for an experiment is kept, so a subscriber that connects
late first receives everything so far, then live events, with no gaps
(history copy and registration happen under one lock).
"""
import asyncio
import threading
from typing import Any, AsyncIterator

_CLOSED = object()


class _Channel:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.closed = False
        self.subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []


class EventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._channels: dict[str, _Channel] = {}

    def open(self, eid: str) -> None:
        with self._lock:
            self._channels[eid] = _Channel()

    def known(self, eid: str) -> bool:
        with self._lock:
            return eid in self._channels

    def history(self, eid: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._channels[eid].events)

    def forget(self, eid: str) -> None:
        with self._lock:
            self._channels.pop(eid, None)

    def publish(self, eid: str, event: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            ch = self._channels[eid]
            event = {**event, "seq": len(ch.events)}
            ch.events.append(event)
            subs = list(ch.subscribers)
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, event)
        return event

    def close(self, eid: str) -> None:
        with self._lock:
            ch = self._channels[eid]
            ch.closed = True
            subs, ch.subscribers = ch.subscribers, []
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, _CLOSED)

    async def subscribe(self, eid: str, loop: asyncio.AbstractEventLoop) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            ch = self._channels[eid]
            backlog, closed = list(ch.events), ch.closed
            if not closed:
                ch.subscribers.append((loop, q))
        for ev in backlog:
            yield ev
        if closed:
            return
        while (ev := await q.get()) is not _CLOSED:
            yield ev
