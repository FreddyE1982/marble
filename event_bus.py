from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List, Literal, Tuple


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

    def publish(self, name: str, data: dict | None = None) -> dict:
        """Publish an event to all subscribers and return the event payload."""
        if not self._subscribers and data is None:
            return {"name": name, "timestamp": time.time()}
        event = standardise_event(name, data)
        for sub in self._subscribers:
            filt = sub["filter"]
            if filt and name not in filt:
                continue
            rate = sub["rate"]
            if rate and event["timestamp"] - sub["last"] < 1.0 / rate:
                continue
            sub["last"] = event["timestamp"]
            try:
                sub["callback"](name, event)
            except Exception:
                # Swallow subscriber errors to avoid disrupting the main flow
                pass
        return event


# ---------------------------------------------------------------------------
# Progress event schema


@dataclass
class ProgressEvent:
    """Structured payload describing pipeline progress.

    Attributes
    ----------
    step: str
        Name of the step currently being processed.
    index: int
        Zero-based index of the step in the pipeline.
    total: int
        Total number of executable steps in the pipeline.
    device: str
        Device on which the step executes (``"cpu"`` or ``"cuda"``).
    status: Literal["started", "completed"]
        Indicates whether the step just began or finished.
    """

    step: str
    index: int
    total: int
    device: str
    status: Literal["started", "completed"]

    def as_dict(self) -> dict:
        """Return the event as a regular dictionary for publishing."""
        return asdict(self)


@dataclass
class Event:
    """Unified event schema for pipeline and dataset logs."""

    name: str
    timestamp: float
    payload: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "timestamp": self.timestamp, **self.payload}


def standardise_event(name: str, data: dict | None = None) -> dict:
    """Return a schema-compliant event dictionary."""

    ev = Event(name, time.time(), data or {})
    return ev.to_dict()


def convert_legacy_events(records: Iterable[Tuple[str, dict]]) -> List[dict]:
    """Convert legacy ``(name, payload)`` records to the unified schema."""

    return [standardise_event(name, payload) for name, payload in records]


# Event name used for pipeline progress notifications
PROGRESS_EVENT = "pipeline_progress"

# Global bus used across the project
global_event_bus = EventBus()
