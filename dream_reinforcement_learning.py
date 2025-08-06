import time

from marble_core import Core, perform_message_passing
from marble_imports import *  # noqa: F403
from marble_neuronenblitz import Neuronenblitz


class DreamReinforcementLearner:
    """Hybrid reinforcement paradigm mixing real and dreamed experiences."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        dream_cycles: int = 1,
        dream_strength: float = 0.5,
        dream_interval: int = 1,
        dream_cycle_duration: float | None = None,
    ) -> None:
        self.core = core
        self.nb = nb
        self.dream_cycles = int(dream_cycles)
        self.dream_strength = float(dream_strength)
        # ``dream_interval`` controls after how many wander cycles dream steps run.
        # Value must be at least one.
        self.dream_interval = max(1, int(dream_interval))
        # Optional fixed duration (in seconds) for each dream step. ``None`` uses
        # the natural execution time without extra delay.
        self.dream_cycle_duration = dream_cycle_duration
        self.history: list[dict] = []
        self._wander_count = 0

    def _dream_step(self, value: float) -> None:
        dream_output, path = self.nb.dynamic_wander(value)
        error = value - dream_output
        self.nb.apply_weight_updates_and_attention(path, error * self.dream_strength)
        perform_message_passing(self.core)
        if self.dream_cycle_duration is not None:
            time.sleep(self.dream_cycle_duration)

    def train_episode(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        self._wander_count += 1
        if self._wander_count % self.dream_interval == 0:
            for _ in range(self.dream_cycles):
                self._dream_step(output)
        self.history.append({"error": float(error)})
        return float(error)

    def train(self, episodes: list[tuple[float, float]], repeat: int = 1) -> None:
        for _ in range(int(repeat)):
            for inp, tgt in episodes:
                self.train_episode(float(inp), float(tgt))
