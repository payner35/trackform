"""In-process event bus.

The Analyzer pipeline publishes `analyzer.*` events here (track_queued,
track_started, track_stage, track_done, track_failed). WebSocket connections
in `server.py` subscribe and receive every published event as JSON.

Single-process service, so a global asyncio queue is fine. When the
Service ever needs cross-process events (multiple uvicorn workers,
horizontal scale), swap this implementation for Redis pub/sub without
changing the publisher or subscriber API.

REALTIME_SPEC §4.2 is the source of truth for event payloads.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventBus:
    """One-process pub/sub. Subscribers each get their own asyncio queue."""

    _subscribers: list[asyncio.Queue[str]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def publish(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Publish synchronously — safe to call from worker threads.

        Each subscriber's queue gets a JSON-serialised copy. Full queues
        drop oldest to make room — the realtime stream is best-effort.
        """
        msg = json.dumps(
            {
                "type": event_type,
                "v": 1,
                "ts_ms": int(time.time() * 1000),
                **(payload or {}),
            }
        )
        for q in list(self._subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except Exception:
                    pass

    async def stream(self, q: asyncio.Queue[str]) -> AsyncIterator[str]:
        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            self.unsubscribe(q)


# Module-level singleton — single-process service, so this is fine.
event_bus = EventBus()
