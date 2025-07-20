from marble_imports import *
from marble_core import perform_message_passing, Core
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marble_neuronenblitz import Neuronenblitz

class ImitationLearner:
    """Behaviour cloning using MARBLE Core and Neuronenblitz."""

    def __init__(self, core: Core, nb: 'Neuronenblitz', max_history: int = 100) -> None:
        self.core = core
        self.nb = nb
        self.max_history = int(max_history)
        self.dataset: list[tuple[float, float]] = []
        self.history: list[dict] = []

    def record(self, input_value: float, action: float) -> None:
        """Store a demonstration pair for later training."""
        self.dataset.append((float(input_value), float(action)))
        if len(self.dataset) > self.max_history:
            self.dataset.pop(0)

    def train_step(self, input_value: float, action: float) -> float:
        """Perform one behaviour cloning update for a single pair."""
        output, path = self.nb.dynamic_wander(float(input_value))
        error = float(action) - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        loss = float(error * error)
        self.history.append({"loss": loss})
        return loss

    def train(self, epochs: int = 1) -> None:
        """Train over all recorded demonstrations."""
        if not self.dataset:
            return
        for _ in range(int(epochs)):
            for inp, act in list(self.dataset):
                self.train_step(inp, act)
