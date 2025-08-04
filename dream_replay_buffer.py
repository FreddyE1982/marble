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
    """Store past experiences for memory consolidation during dreams."""

    def __init__(self, capacity: int, weighting: str = "linear") -> None:
        self.capacity = int(capacity)
        self.weighting = weighting
        self.buffer: Deque[DreamExperience] = deque()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    # ------------------------------------------------------------------
    def add(self, exp: DreamExperience) -> None:
        """Add an experience with salience-based eviction."""

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            return
        # Evict lowest salience item if buffer full
        idx, min_exp = min(enumerate(self.buffer), key=lambda x: x[1].salience)
        if exp.salience > min_exp.salience:
            self.buffer[idx] = exp
        # else: drop new experience

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> List[DreamExperience]:
        """Sample a batch biased toward high-salience experiences."""

        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        saliences = np.array([e.salience for e in self.buffer], dtype=float)
        if self.weighting == "exponential":
            saliences = np.exp(saliences)
        probs = saliences / saliences.sum()
        idx = np.random.choice(
            len(self.buffer), size=batch_size, replace=False, p=probs
        )
        return [self.buffer[i] for i in idx.tolist()]
