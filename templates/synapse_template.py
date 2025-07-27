"""Template for custom synapse types."""

from marble_core import Synapse


def create_custom_synapse(source: int, target: int, weight: float = 1.0) -> Synapse:
    """Return a synapse with example parameters."""
    syn = Synapse(source=source, target=target, weight=weight, synapse_type="standard")
    syn.metadata = {"description": "Replace with custom synapse dynamics"}
    return syn
