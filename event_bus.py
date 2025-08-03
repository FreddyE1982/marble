from __future__ import annotations
import time
from typing import Callable, Iterable

class EventBus:
    """Publish/subscribe system for MARBLE events.

    Subscribers can filter by event name and apply a rate limit to reduce
    overhead. Callbacks receive the event name and an associated payload
    dictionary.
    """

    def __init__(self) -> None:
        self._subscribers: list[dict] = []

    def subscribe(
        self,
        callback: Callable[[str, dict], None],
        *,
        events: Iterable[str] | None = None,
        rate_limit_hz: float | None = None,
    ) -> None:
        """Register a callback for events.

        Args:
            callback: Function invoked with ``(name, data)`` for each event.
            events: Optional iterable of event names to receive. ``None``
                subscribes to all events.
            rate_limit_hz: Maximum number of events per second delivered to this
                subscriber. ``None`` disables rate limiting.
        """
        filt = set(events) if events else None
        self._subscribers.append(
            {"callback": callback, "filter": filt, "rate": rate_limit_hz, "last": 0.0}
        )

    def publish(self, name: str, data: dict | None = None) -> None:
        """Publish an event to all subscribers."""
        if not self._subscribers:
            return
        payload = data or {}
        now = time.time()
        for sub in self._subscribers:
            filt = sub["filter"]
            if filt and name not in filt:
                continue
            rate = sub["rate"]
            if rate and now - sub["last"] < 1.0 / rate:
                continue
            sub["last"] = now
            try:
                sub["callback"](name, payload)
            except Exception:
                # Swallow subscriber errors to avoid disrupting the main flow
                pass


# Global bus used across the project
global_event_bus = EventBus()
