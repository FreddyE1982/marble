from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz
import math

class QuantumFluxLearner:
    """Imaginary phase-based learning rule using per-synapse flux."""

    def __init__(self, core: Core, nb: Neuronenblitz, phase_rate: float = 0.1) -> None:
        self.core = core
        self.nb = nb
        self.phase_rate = float(phase_rate)
        self.phases: dict = {}
        self.history: list[dict] = []

    def _get_phase(self, syn) -> float:
        return self.phases.setdefault(syn, 0.0)

    def train_step(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        for syn in path:
            phase = self._get_phase(syn)
            amplitude = math.sin(phase)
            delta = self.nb.weight_update_fn(
                self.core.neurons[syn.source].value,
                error * amplitude,
                len(path),
            )
            if abs(delta) > self.nb.synapse_update_cap:
                delta = math.copysign(self.nb.synapse_update_cap, delta)
            syn.weight += self.nb.learning_rate * delta
            self.phases[syn] = phase + self.phase_rate * float(error)
        perform_message_passing(self.core)
        self.history.append({"error": float(error)})
        return float(error)

    def train(self, examples: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp, tgt in examples:
                self.train_step(float(inp), float(tgt))
