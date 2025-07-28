from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Deque, List


@dataclass
class BroadcastMessage:
    """Message broadcast through the Global Workspace."""

    source: str
    content: Any
    timestamp: float = field(default_factory=time)


class GlobalWorkspace:
    """Central message queue shared between plugins and components."""

    def __init__(self, capacity: int = 100) -> None:
        self.queue: Deque[BroadcastMessage] = deque(maxlen=capacity)
        self.subscribers: List[Callable[[BroadcastMessage], None]] = []

    def publish(self, source: str, content: Any) -> None:
        """Add a message and notify all subscribers."""
        msg = BroadcastMessage(source, content)
        self.queue.append(msg)
        for cb in list(self.subscribers):
            cb(msg)

    def subscribe(self, callback: Callable[[BroadcastMessage], None]) -> None:
        """Register ``callback`` to receive future messages."""
        self.subscribers.append(callback)


# Global instance created on activation
workspace: GlobalWorkspace | None = None


def activate(nb: object | None = None, capacity: int = 100) -> GlobalWorkspace:
    """Initialise the global workspace and attach to ``nb`` if given."""
    global workspace
    if workspace is None or workspace.queue.maxlen != capacity:
        workspace = GlobalWorkspace(capacity)
    if nb is not None:
        setattr(nb, "global_workspace", workspace)
    return workspace


def register(*_: Callable[[str], None]) -> None:
    """No neuron or synapse types are registered by this plugin."""
    return
