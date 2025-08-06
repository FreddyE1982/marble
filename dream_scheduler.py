from __future__ import annotations

import time
from typing import Callable, Iterable

from dream_replay_buffer import DreamReplayBuffer, DreamExperience
from marble_neuronenblitz import Neuronenblitz


class DreamScheduler:
    """Coordinate dream replay, weighting and housekeeping.

    The scheduler samples high-salience :class:`DreamExperience` instances from a
    :class:`DreamReplayBuffer` and replays them through a
    :class:`~marble_neuronenblitz.Neuronenblitz` learner.  After each replay
    cycle the buffer's housekeeping step removes memories that fall below the
    configured salience threshold.  This orchestration ensures replay, weighting
    and pruning run together in a single call.

    Parameters
    ----------
    nb:
        ``Neuronenblitz`` instance used for training.
    buffer:
        ``DreamReplayBuffer`` providing stored experiences.
    batch_size:
        Number of experiences to sample per cycle.
    """

    def __init__(self, nb: Neuronenblitz, buffer: DreamReplayBuffer, batch_size: int) -> None:
        self.nb = nb
        self.buffer = buffer
        self.batch_size = int(batch_size)

    # ------------------------------------------------------------------
    def replay(self) -> int:
        """Run a single replay cycle.

        Returns the number of experiences that were replayed.  The buffer's
        weighting policy determines which memories are sampled.
        """

        batch: Iterable[DreamExperience] = self.buffer.sample(self.batch_size)
        for exp in batch:
            self.nb.train_example(exp.input_value, exp.target_value)
        # ``sample`` already performs housekeeping but invoking it again keeps
        # the contract explicit and allows external uses of ``replay`` to rely
        # on pruning having run.
        self.buffer.housekeeping()
        return len(list(batch))

    # ------------------------------------------------------------------
    def run(self, cycles: int, *, interval: float | int = 0) -> None:
        """Execute ``cycles`` replay iterations with optional sleep ``interval``."""

        for _ in range(int(cycles)):
            self.replay()
            if interval:
                time.sleep(interval)
