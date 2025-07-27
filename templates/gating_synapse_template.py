"""Reusable template for a gating synapse."""

from marble_core import Synapse

class GatingSynapse(Synapse):
    """Synapse that modulates transmission by a gate value."""

    def __init__(self, source: int, target: int, weight: float = 1.0, gate: float = 1.0):
        super().__init__(source=source, target=target, weight=weight, synapse_type="gating")
        self.gate = gate

    def transmit(self, value: float) -> float:
        """Return gated value."""
        return value * self.gate
