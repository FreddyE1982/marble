"""Reusable template for an RNN-style neuron."""

from marble_core import Neuron

class RNNNeuron(Neuron):
    """A neuron with internal state for simple recurrent networks."""

    def __init__(self, neuron_id: int, rep_size: int = 4, hidden_state: float = 0.0):
        super().__init__(neuron_id, value=0.0, tier="vram", rep_size=rep_size)
        self.hidden_state = hidden_state

    def step(self, input_value: float) -> float:
        """Update internal state and return output."""
        self.hidden_state = 0.5 * self.hidden_state + input_value
        return self.hidden_state
