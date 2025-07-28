"""Memory management utilities for ``Neuronenblitz``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import Neuronenblitz


def decay_memory_gates(nb: "Neuronenblitz") -> None:
    """Decay memory gate strengths over time."""
    for syn in list(nb.memory_gates.keys()):
        nb.memory_gates[syn] *= nb.memory_gate_decay
        if nb.memory_gates[syn] < 1e-6:
            del nb.memory_gates[syn]
