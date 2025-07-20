from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz

class CurriculumLearner:
    """Progressively trains on examples ordered by difficulty."""

    def __init__(self, core: Core, nb: Neuronenblitz, difficulty_fn=None, schedule: str = "linear") -> None:
        self.core = core
        self.nb = nb
        self.difficulty_fn = difficulty_fn if difficulty_fn is not None else (lambda pair: 0.0)
        self.schedule = schedule
        self.history: list[dict] = []

    def _curriculum_threshold(self, epoch: int, total_epochs: int, n_samples: int) -> int:
        if self.schedule == "exponential":
            ratio = 1.0 - math.exp(-epoch)
            max_ratio = 1.0 - math.exp(-total_epochs)
            ratio /= max_ratio if max_ratio > 0 else 1.0
        else:
            ratio = epoch / total_epochs
        return max(1, int(n_samples * ratio))

    def train(self, dataset: list[tuple[float, float]], epochs: int = 1) -> list[float]:
        data = list(dataset)
        diffs = [self.difficulty_fn(p) for p in data]
        ordered = [x for _, x in sorted(zip(diffs, data), key=lambda t: t[0])]
        losses = []
        n = len(ordered)
        for ep in range(1, int(epochs) + 1):
            upto = self._curriculum_threshold(ep, int(epochs), n)
            subset = ordered[:upto]
            for inp, target in subset:
                out, path = self.nb.dynamic_wander(inp)
                error = target - out
                self.nb.apply_weight_updates_and_attention(path, error)
                perform_message_passing(self.core)
                loss = float(error * error)
                losses.append(loss)
                self.history.append({"input": float(inp), "target": float(target), "loss": loss})
        return losses
