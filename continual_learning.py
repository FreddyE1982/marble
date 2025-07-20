from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz
import random

class ReplayContinualLearner:
    """Experience replay based continual learning integrated with MARBLE."""

    def __init__(self, core: Core, nb: Neuronenblitz, memory_size: int = 10) -> None:
        self.core = core
        self.nb = nb
        self.memory_size = int(memory_size)
        self.memory: list[tuple[float, float]] = []
        self.history: list[dict] = []

    def _store_example(self, pair: tuple[float, float]) -> None:
        if len(self.memory) < self.memory_size:
            self.memory.append(pair)
        else:
            idx = random.randrange(self.memory_size)
            self.memory[idx] = pair

    def _replay(self) -> None:
        if not self.memory:
            return
        inp, target = random.choice(self.memory)
        out, path = self.nb.dynamic_wander(inp)
        error = target - out
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)

    def train_step(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        self._store_example((input_value, target_value))
        self._replay()
        loss = float(error * error)
        self.history.append({"loss": loss})
        return loss

    def train(self, examples: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp, tgt in examples:
                self.train_step(float(inp), float(tgt))
