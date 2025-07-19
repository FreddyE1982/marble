import math
from typing import Iterable, Tuple, Any
from marble_brain import Brain, _normalize_examples
from tqdm import tqdm


class DistillationTrainer:
    """Train a student brain using a teacher brain for guidance."""

    def __init__(self, student: Brain, teacher: Brain, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.student = student
        self.teacher = teacher
        self.alpha = alpha

    def _blend_targets(
        self, examples: Iterable[Tuple[float, float]]
    ) -> list[Tuple[float, float]]:
        blended = []
        for inp, tgt in _normalize_examples(examples):
            teacher_out = self.teacher.infer(inp)
            mixed = (1.0 - self.alpha) * tgt + self.alpha * teacher_out
            blended.append((inp, mixed))
        return blended

    def train(
        self,
        train_examples: Iterable[Any],
        epochs: int = 1,
        validation_examples: Iterable[Any] | None = None,
    ) -> None:
        """Train ``self.student`` using teacher-guided targets."""
        blended = self._blend_targets(train_examples)
        pbar = tqdm(range(epochs), desc="DistillationEpochs", ncols=100)
        for _ in pbar:
            self.student.train(blended, epochs=1, validation_examples=validation_examples)
        pbar.close()
