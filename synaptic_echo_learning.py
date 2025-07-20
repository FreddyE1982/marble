from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz

class SynapticEchoLearner:
    """Experimental learning paradigm using synaptic echo buffers."""

    def __init__(self, core: Core, nb: Neuronenblitz, echo_influence: float = 1.0) -> None:
        self.core = core
        self.nb = nb
        self.echo_influence = float(echo_influence)
        self.nb.use_echo_modulation = True
        self.history: list[dict] = []

    def train_step(self, value: float, target: float) -> float:
        out, path = self.nb.dynamic_wander(value)
        error = target - out
        self.nb.apply_weight_updates_and_attention(path, error * self.echo_influence)
        perform_message_passing(self.core)
        self.history.append({"input": value, "target": target, "error": float(error)})
        return float(error)

    def train(self, pairs: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp, tgt in pairs:
                self.train_step(float(inp), float(tgt))
