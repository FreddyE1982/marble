"""Reinforcement learning helpers for :class:`Neuronenblitz`.

These functions operate on a ``Neuronenblitz`` instance and are kept
in a separate module to keep :mod:`core` focused on graph dynamics.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import Neuronenblitz

import numpy as np


def enable_rl(nb: "Neuronenblitz") -> None:
    """Enable built-in reinforcement learning."""
    nb.rl_enabled = True


def disable_rl(nb: "Neuronenblitz") -> None:
    """Disable built-in reinforcement learning."""
    nb.rl_enabled = False


def rl_select_action(
    nb: "Neuronenblitz", state: Tuple[int, int], n_actions: int
) -> int:
    """Return an action using epsilon-greedy selection."""
    if not nb.rl_enabled:
        raise RuntimeError("reinforcement learning disabled")
    if random.random() < nb.rl_epsilon:
        return random.randrange(n_actions)
    q_vals = [nb.dynamic_wander(nb.q_encoding(state, a))[0] for a in range(n_actions)]
    return int(np.argmax(q_vals))


def rl_update(
    nb: "Neuronenblitz",
    state: Tuple[int, int],
    action: int,
    reward: float,
    next_state: Tuple[int, int],
    done: bool,
    n_actions: int = 4,
) -> None:
    """Perform a Q-learning update using ``dynamic_wander``."""
    if not nb.rl_enabled:
        return
    next_q = 0.0
    if not done:
        next_q = max(
            nb.dynamic_wander(nb.q_encoding(next_state, a))[0] for a in range(n_actions)
        )
    target = reward + nb.rl_discount * next_q
    nb.train([(nb.q_encoding(state, action), target)], epochs=1)
    nb.rl_epsilon = max(nb.rl_min_epsilon, nb.rl_epsilon * nb.rl_epsilon_decay)
