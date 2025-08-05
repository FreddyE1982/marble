"""Plugin interface for attention codelets and coalition formation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import torch

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
_workspace_gates: dict[str, float] = {}


def register_codelet(func: Callable[[], AttentionProposal]) -> None:
    """Register an attention codelet callback.

    Args:
        func: Callable that returns an :class:`AttentionProposal` when invoked.
    """
    _codelets.append(func)


def get_codelets() -> list[Callable[[], AttentionProposal]]:
    """Return all registered codelet callbacks."""
    return list(_codelets)


def _workspace_listener(msg: global_workspace.BroadcastMessage) -> None:
    if not isinstance(msg.content, dict):
        return
    name = msg.content.get("codelet")
    gate = msg.content.get("gate")
    if name is None or gate is None:
        return
    try:
        _workspace_gates[name] = float(gate)
    except Exception:  # pragma: no cover - defensive
        return


def enable_workspace_gating() -> None:
    """Subscribe to Global Workspace events to adjust codelet salience."""

    if global_workspace.workspace is None:
        global_workspace.activate()
    global_workspace.workspace.subscribe(_workspace_listener)


def form_coalition(
    coalition_size: int | None = None,
    *,
    saliences: list[float] | None = None,
    salience_weight: float | None = None,
    device: torch.device | None = None,
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proposals = [codelet() for codelet in _codelets]
    if not proposals:
        return []
    scores = torch.tensor(
        [p.score for p in proposals], dtype=torch.float32, device=device
    )
    if saliences is not None:
        sarr = torch.tensor(saliences, dtype=torch.float32, device=device)
        scores = scores + salience_weight * sarr
    if _workspace_gates:
        gates = torch.tensor(
            [
                _workspace_gates.get(getattr(codelet, "__name__", str(i)), 0.0)
                for i, codelet in enumerate(_codelets)
            ],
            dtype=torch.float32,
            device=device,
        )
        scores = scores + gates
    probs = torch.softmax(scores, dim=0)
    idx = torch.argsort(probs, descending=True)[:coalition_size]
    return [proposals[i] for i in idx.tolist()]


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


def run_cycle(
    coalition_size: int | None = None, *, device: torch.device | None = None
) -> None:
    """Form a coalition and broadcast the winners."""

    coalition = form_coalition(coalition_size, device=device)
    if global_workspace.workspace is not None:
        for proposal in coalition:
            global_workspace.workspace.publish("attention_codelets", proposal.content)
    else:
        broadcast_coalition(coalition)


def activate(*, coalition_size: int = 1, salience_weight: float = 1.0) -> None:
    """Activate the attention codelet subsystem.

    Args:
        coalition_size: Number of proposals broadcast per cycle.
    """

    global _default_coalition_size, _salience_weight
    _default_coalition_size = coalition_size
    _salience_weight = salience_weight
