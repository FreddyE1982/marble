"""Global Workspace plugin used by consciousness modules."""

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
        """Add a message and notify all subscribers.

        Args:
            source: Identifier of the sender.
            content: Arbitrary payload to deliver.
        """
        msg = BroadcastMessage(source, content)
        self.queue.append(msg)
        for cb in list(self.subscribers):
            cb(msg)

    def subscribe(self, callback: Callable[[BroadcastMessage], None]) -> None:
        """Register ``callback`` to receive future messages.

        Args:
            callback: Function invoked with each :class:`BroadcastMessage`.
        """
        self.subscribers.append(callback)


# Global instance created on activation
workspace: GlobalWorkspace | None = None


def activate(nb: object | None = None, capacity: int = 100) -> GlobalWorkspace:
    """Initialise the global workspace and optionally attach it to ``nb``.

    Args:
        nb: Object to attach the workspace to. If ``None`` the workspace is
            created but not assigned.
        capacity: Maximum number of messages stored.

    Returns:
        The shared :class:`GlobalWorkspace` instance.
    """
    global workspace
    if workspace is None or workspace.queue.maxlen != capacity:
        workspace = GlobalWorkspace(capacity)
    if nb is not None:
        setattr(nb, "global_workspace", workspace)
    return workspace


def register(*_: Callable[[str], None]) -> None:
    """Required plugin entry point.

    The global workspace does not register custom neuron or synapse types,
    but this function is provided for consistency with the plugin interface.
    """
    return
