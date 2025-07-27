from marble_core import Core, perform_message_passing
from marble_imports import math
from marble_neuronenblitz import Neuronenblitz


class CurriculumLearner:
    """Progressively trains on examples ordered by difficulty."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        difficulty_fn=None,
        schedule: str = "linear",
    ) -> None:
        self.core = core
        self.nb = nb
        self.difficulty_fn = (
            difficulty_fn if difficulty_fn is not None else (lambda pair: 0.0)
        )
        self.schedule = schedule
        self.history: list[dict] = []

    def _curriculum_threshold(
        self, epoch: int, total_epochs: int, n_samples: int
    ) -> int:
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
                self.history.append(
                    {"input": float(inp), "target": float(target), "loss": loss}
                )
        return losses


def curriculum_train(
    core: Core,
    nb: Neuronenblitz,
    dataset: list[tuple[float, float]],
    *,
    epochs: int = 1,
    difficulty_fn=None,
    schedule: str = "linear",
) -> list[float]:
    """Train ``nb`` on ``dataset`` using curriculum learning.

    Parameters
    ----------
    core:
        Target :class:`~marble_core.Core` instance.
    nb:
        :class:`~marble_neuronenblitz.Neuronenblitz` associated with ``core``.
    dataset:
        Sequence of ``(input, target)`` pairs.
    epochs:
        Number of training epochs.
    difficulty_fn:
        Optional function mapping ``(input, target)`` to a difficulty score.
    schedule:
        ``"linear"`` or ``"exponential"`` progression schedule.

    Returns
    -------
    list[float]
        Recorded loss values for each training sample.
    """

    learner = CurriculumLearner(
        core, nb, difficulty_fn=difficulty_fn, schedule=schedule
    )
    return learner.train(dataset, epochs=epochs)
