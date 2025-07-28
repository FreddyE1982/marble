"""Plugin interface for attention codelets and coalition formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import numpy as np

import global_workspace


@dataclass
class AttentionProposal:
    """Proposal returned by a codelet.

    Parameters
    ----------
    score:
        Salience score of the proposal. Higher scores are more likely to win the
        coalition.
    content:
        Arbitrary object representing the information to broadcast when this
        proposal is selected.
    """

    score: float
    content: Any


_codelets: List[Callable[[], AttentionProposal]] = []
_default_coalition_size = 1
_salience_weight = 1.0


def register_codelet(func: Callable[[], AttentionProposal]) -> None:
    """Register an attention codelet callback.

    Args:
        func: Callable that returns an :class:`AttentionProposal` when invoked.
    """
    _codelets.append(func)


def get_codelets() -> list[Callable[[], AttentionProposal]]:
    """Return all registered codelet callbacks."""
    return list(_codelets)


def form_coalition(
    coalition_size: int | None = None,
    *,
    saliences: list[float] | None = None,
    salience_weight: float | None = None,
) -> list[AttentionProposal]:
    """Return the highest scoring proposals.

    Args:
        coalition_size: Number of proposals to return. When ``None`` the
            default from :func:`activate` is used.

    Returns:
        The winning proposals sorted by score.
    """

    if coalition_size is None:
        coalition_size = _default_coalition_size
    if salience_weight is None:
        salience_weight = _salience_weight
    proposals = [codelet() for codelet in _codelets]
    if not proposals:
        return []
    scores = np.array([p.score for p in proposals], dtype=float)
    if saliences is not None:
        sarr = np.array(saliences, dtype=float)
        scores = scores + salience_weight * sarr
    probs = np.exp(scores) / np.sum(np.exp(scores))
    idx = np.argsort(probs)[-coalition_size:][::-1]
    return [proposals[i] for i in idx]


def broadcast_coalition(coalition: list[AttentionProposal]) -> None:
    """Broadcast proposals via the global workspace if available.

    Args:
        coalition: Proposals returned by :func:`form_coalition`.
    """

    if global_workspace.workspace is None:
        return
    for proposal in coalition:
        msg = {"content": proposal.content, "score": proposal.score}
        global_workspace.workspace.publish("attention_codelets", msg)


def run_cycle(coalition_size: int | None = None) -> None:
    """Form a coalition and broadcast the winners."""

    coalition = form_coalition(coalition_size)
    broadcast_coalition(coalition)


def activate(*, coalition_size: int = 1, salience_weight: float = 1.0) -> None:
    """Activate the attention codelet subsystem.

    Args:
        coalition_size: Number of proposals broadcast per cycle.
    """

    global _default_coalition_size, _salience_weight
    _default_coalition_size = coalition_size
    _salience_weight = salience_weight
