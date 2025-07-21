from marble_imports import *
from marble_core import Core, perform_message_passing, Neuron
from marble_neuronenblitz import Neuronenblitz
import random
import numpy as np

class ConceptualIntegrationLearner:
    """Rule-based paradigm creating blended concept neurons."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        blend_probability: float = 0.1,
        similarity_threshold: float = 0.3,
    ) -> None:
        self.core = core
        self.nb = nb
        self.blend_probability = float(blend_probability)
        self.similarity_threshold = float(similarity_threshold)
        # disable weight learning
        self.nb.learning_rate = 0.0
        self.nb.weight_decay = 0.0

    def _blend(self, i: int, j: int) -> None:
        rep_i = self.core.neurons[i].representation
        rep_j = self.core.neurons[j].representation
        new_rep = np.tanh(rep_i * rep_j)
        new_id = len(self.core.neurons)
        tier = self.core.choose_new_tier()
        neuron = Neuron(new_id, value=0.0, tier=tier, rep_size=self.core.rep_size)
        neuron.representation = new_rep.astype(np.float32)
        self.core.neurons.append(neuron)
        self.core.add_synapse(new_id, i, weight=1.0)
        self.core.add_synapse(new_id, j, weight=1.0)
        self.core.add_synapse(i, new_id, weight=1.0)
        self.core.add_synapse(j, new_id, weight=1.0)

    def _maybe_blend(self, active: list[int]) -> None:
        if len(active) < 2:
            return
        if random.random() > self.blend_probability:
            return
        i, j = random.sample(active, 2)
        ri = self.core.neurons[i].representation
        rj = self.core.neurons[j].representation
        denom = max(np.linalg.norm(ri) * np.linalg.norm(rj), 1e-8)
        cos = float(np.dot(ri, rj) / denom)
        if abs(cos) > self.similarity_threshold:
            return
        self._blend(i, j)

    def train_step(self, value: float) -> float:
        out, path = self.nb.dynamic_wander(float(value), apply_plasticity=False)
        active = [syn.target for syn in path]
        if path:
            active.append(path[0].source)
        self._maybe_blend(active)
        perform_message_passing(self.core)
        return float(out) if isinstance(out, (int, float)) else float(np.mean(out))

    def train(self, inputs: list[float], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for val in inputs:
                self.train_step(float(val))
