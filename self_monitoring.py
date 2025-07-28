"""Self-Monitoring plugin for meta-cognitive state tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict

import global_workspace


@dataclass
class MonitorState:
    """Internal state captured by the self-monitoring plugin."""

    error_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    mean_error: float = 0.0


class SelfMonitor:
    """Collects errors and emits meta-cognitive markers."""

    def __init__(self, nb: object, history_size: int = 100) -> None:
        self.nb = nb
        self.state = MonitorState(deque(maxlen=history_size))

    def update_error(self, error: float) -> None:
        """Record ``error`` and update running statistics."""
        self.state.error_history.append(float(error))
        self.state.mean_error = sum(self.state.error_history) / len(self.state.error_history)
        marker = {"mean_error": self.state.mean_error}
        if hasattr(self.nb, "log_hot_marker"):
            self.nb.log_hot_marker(marker)
        if global_workspace.workspace is not None:
            global_workspace.workspace.publish("self_monitoring", marker)


_monitor: SelfMonitor | None = None


def activate(nb: object, history_size: int = 100) -> None:
    """Attach the self-monitoring plugin to ``nb``."""
    global _monitor
    _monitor = SelfMonitor(nb, history_size)
    setattr(nb, "self_monitor", _monitor)


def log_error(error: float) -> None:
    """Public helper to update the active monitor if present."""
    if _monitor is not None:
        _monitor.update_error(error)


def register(*_: Any) -> None:
    """Required for plugin loader compatibility."""
    return
