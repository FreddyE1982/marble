from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class HebbianLearner:
    """Unsupervised Hebbian learning integrated with MARBLE."""

    def __init__(self, core: Core, nb: Neuronenblitz, learning_rate: float = 0.01, weight_decay: float = 0.0) -> None:
        self.core = core
        self.nb = nb
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.history: list[dict] = []
        self._weight_limit = 1e6

    def _update_weights(self, path: list) -> None:
        for syn in path:
            pre = self.core.neurons[syn.source].value
            post = self.core.neurons[syn.target].value
            if pre is None or post is None:
                continue
            delta = self.learning_rate * float(pre) * float(post)
            delta -= self.weight_decay * syn.weight
            syn.weight += delta * getattr(self.nb, "plasticity_modulation", 1.0)
            if syn.weight > self._weight_limit:
                syn.weight = self._weight_limit
            elif syn.weight < -self._weight_limit:
                syn.weight = -self._weight_limit

    def train_step(self, input_value: float) -> float:
        out, path = self.nb.dynamic_wander(input_value, apply_plasticity=False)
        perform_message_passing(self.core)
        self._update_weights(path)
        self.history.append({"input": input_value, "output": out, "path_len": len(path)})
        return float(out) if isinstance(out, (int, float)) else float(np.mean(out))

    def train(self, inputs: list, epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp in inputs:
                self.train_step(inp)
