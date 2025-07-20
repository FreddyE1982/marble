from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz

class HarmonicResonanceLearner:
    """Novel frequency-based learning paradigm."""

    def __init__(self, core: Core, nb: Neuronenblitz, base_frequency: float = 1.0, decay: float = 0.99) -> None:
        self.core = core
        self.nb = nb
        self.base_frequency = float(base_frequency)
        self.decay = float(decay)
        self.history: list[dict] = []

    def _encode(self, value: float) -> cp.ndarray:
        freq = self.base_frequency
        rep = cp.zeros(self.core.rep_size, dtype=float)
        rep[0] = cp.sin(freq * value)
        if self.core.rep_size > 1:
            rep[1] = cp.cos(freq * value)
        return rep

    def train_step(self, value: float, target: float) -> float:
        rep = self._encode(value)
        if CUDA_AVAILABLE:
            self.core.neurons[0].representation = cp.asnumpy(rep)
        else:
            self.core.neurons[0].representation = cp.asnumpy(rep)
        output, path = self.nb.dynamic_wander(value)
        error = float(target - output)
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        self.base_frequency *= self.decay
        self.history.append({"input": value, "target": target, "error": error})
        return error
