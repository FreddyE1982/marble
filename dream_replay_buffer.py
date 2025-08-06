from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Deque, List

import numpy as np


@dataclass
class DreamExperience:
    """Single training experience stored for dream replay."""

    input_value: float
    target_value: float
    reward: float
    emotion: float
    arousal: float
    stress: float
    timestamp: float = field(default_factory=time)
    salience: float = field(init=False)

    def __post_init__(self) -> None:
        for name, value in {
            "emotion": self.emotion,
            "arousal": self.arousal,
            "stress": self.stress,
            "reward": self.reward,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        # Simple salience heuristic: average of tags and reward
        self.salience = (
            float(self.reward + self.emotion + self.arousal + (1 - self.stress)) / 4.0
        )


class DreamReplayBuffer:
    """Store past experiences for memory consolidation during dreams.

    The buffer consists of a short-term *instant* buffer capturing the most
    recent experiences and a long-term buffer used for consolidated replay. The
    ``merge_instant_buffer`` method moves data from the instant buffer into the
    long-term store where salience-based eviction and optional housekeeping are
    applied.
    """

    def __init__(
        self,
        capacity: int,
        weighting: str = "linear",
        *,
        instant_capacity: int = 10,
        housekeeping_threshold: float = 0.05,
    ) -> None:
        self.capacity = int(capacity)
        self.weighting = weighting
        self.buffer: Deque[DreamExperience] = deque()
        self.instant_capacity = int(instant_capacity)
        self.instant_buffer: Deque[DreamExperience] = deque()
        self.housekeeping_threshold = float(housekeeping_threshold)

    # ------------------------------------------------------------------
    def _apply_weighting(self, saliences: np.ndarray) -> np.ndarray:
        """Return weighted ``saliences`` according to strategy."""

        if self.weighting == "linear":
            return saliences
        if self.weighting == "exponential":
            return np.exp(saliences)
        if self.weighting == "quadratic":
            return saliences**2
        if self.weighting == "sqrt":
            return np.sqrt(saliences)
        if self.weighting == "uniform":
            return np.ones_like(saliences)
        raise ValueError(f"Unknown weighting: {self.weighting}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer) + len(self.instant_buffer)

    # ------------------------------------------------------------------
    def _insert(self, exp: DreamExperience) -> None:
        """Insert ``exp`` into the long-term buffer with salience eviction."""

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            return
        idx, min_exp = min(enumerate(self.buffer), key=lambda x: x[1].salience)
        if exp.salience > min_exp.salience:
            self.buffer[idx] = exp

    def merge_instant_buffer(self) -> None:
        """Merge short-term experiences into the long-term buffer."""

        while self.instant_buffer:
            self._insert(self.instant_buffer.popleft())
        self.housekeeping()

    def housekeeping(self) -> None:
        """Prune experiences below the housekeeping threshold."""

        if self.housekeeping_threshold <= 0:
            return
        self.buffer = deque(
            [e for e in self.buffer if e.salience >= self.housekeeping_threshold],
            maxlen=self.capacity,
        )

    def add(self, exp: DreamExperience) -> None:
        """Add an experience to the instant buffer."""

        self.instant_buffer.append(exp)
        if len(self.instant_buffer) >= self.instant_capacity:
            self.merge_instant_buffer()

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> List[DreamExperience]:
        """Sample a batch biased toward high-salience experiences."""

        self.merge_instant_buffer()
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        saliences = np.array([e.salience for e in self.buffer], dtype=float)
        weights = self._apply_weighting(saliences)
        probs = weights / weights.sum()
        idx = np.random.choice(
            len(self.buffer), size=batch_size, replace=False, p=probs
        )
        return [self.buffer[i] for i in idx.tolist()]
