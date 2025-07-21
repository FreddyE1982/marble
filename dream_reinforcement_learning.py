from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz

class DreamReinforcementLearner:
    """Hybrid reinforcement paradigm mixing real and dreamed experiences."""

    def __init__(self, core: Core, nb: Neuronenblitz, dream_cycles: int = 1, dream_strength: float = 0.5) -> None:
        self.core = core
        self.nb = nb
        self.dream_cycles = int(dream_cycles)
        self.dream_strength = float(dream_strength)
        self.history: list[dict] = []

    def _dream_step(self, value: float) -> None:
        dream_output, path = self.nb.dynamic_wander(value)
        error = value - dream_output
        self.nb.apply_weight_updates_and_attention(path, error * self.dream_strength)
        perform_message_passing(self.core)

    def train_episode(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        for _ in range(self.dream_cycles):
            self._dream_step(output)
        self.history.append({"error": float(error)})
        return float(error)

    def train(self, episodes: list[tuple[float, float]], repeat: int = 1) -> None:
        for _ in range(int(repeat)):
            for inp, tgt in episodes:
                self.train_episode(float(inp), float(tgt))
