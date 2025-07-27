"""Template for custom neuron types."""

import numpy as np
from marble_core import Neuron


def create_custom_neuron(neuron_id: int, rep_size: int = 4) -> Neuron:
    """Return a neuron initialised with zeros and custom metadata."""
    neuron = Neuron(neuron_id, value=0.0, tier="vram", rep_size=rep_size)
    neuron.metadata = {"description": "Replace with custom behaviour"}
    return neuron
