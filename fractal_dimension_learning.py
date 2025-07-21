from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz
from scipy.spatial.distance import pdist
import numpy as np

class FractalDimensionLearner:
    """Adaptive paradigm adjusting representation size via fractal dimension."""

    def __init__(self, core: Core, nb: Neuronenblitz, target_dimension: float = 4.0) -> None:
        self.core = core
        self.nb = nb
        self.target_dimension = float(target_dimension)
        self.history: list[dict] = []

    def _estimate_dimension(self) -> float:
        reps = np.stack([n.representation for n in self.core.neurons])
        if reps.size == 0:
            return 0.0
        dists = pdist(reps)
        if len(dists) == 0:
            return 0.0
        eps = float(np.median(dists))
        counts = float(np.sum(dists < eps))
        if eps <= 0 or counts <= 1:
            return 0.0
        return float(np.log(counts) / np.log(eps))

    def train_step(self, input_value: float, target_value: float) -> float:
        output, path = self.nb.dynamic_wander(input_value)
        error = target_value - output
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        dim = self._estimate_dimension()
        if dim > self.target_dimension:
            self.core.increase_representation_size(1)
            self.target_dimension = self.core.rep_size * 0.8
        self.history.append({"error": float(error), "dimension": dim})
        return float(error)

    def train(self, examples: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for inp, tgt in examples:
                self.train_step(float(inp), float(tgt))
