"""Plugin interface for attention codelets and coalition formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

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


def register_codelet(func: Callable[[], AttentionProposal]) -> None:
    """Register an attention codelet callback."""
    _codelets.append(func)


def get_codelets() -> list[Callable[[], AttentionProposal]]:
    """Return all registered codelet callbacks."""
    return list(_codelets)


def form_coalition(coalition_size: int | None = None) -> list[AttentionProposal]:
    """Return the top proposals from all registered codelets."""

    if coalition_size is None:
        coalition_size = _default_coalition_size
    proposals = [codelet() for codelet in _codelets]
    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals[:coalition_size]


def broadcast_coalition(coalition: list[AttentionProposal]) -> None:
    """Broadcast each proposal to the global workspace if active."""

    if global_workspace.workspace is None:
        return
    for proposal in coalition:
        global_workspace.workspace.publish("attention_codelets", proposal.content)


def run_cycle(coalition_size: int | None = None) -> None:
    """Form a coalition and broadcast the winners."""

    coalition = form_coalition(coalition_size)
    broadcast_coalition(coalition)


def activate(*, coalition_size: int = 1) -> None:
    """Activate the attention codelet subsystem.

    Parameters
    ----------
    coalition_size:
        Number of proposals to broadcast per cycle when :func:`run_cycle` is
        called. Defaults to ``1``.
    """

    global _default_coalition_size
    _default_coalition_size = coalition_size
