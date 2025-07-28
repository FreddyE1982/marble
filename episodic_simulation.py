"""Episodic simulation plugin for planning and replay."""

from __future__ import annotations

from typing import Iterable, List

from marble_neuronenblitz import Neuronenblitz

from episodic_memory import EpisodicEntry, EpisodicMemory


def simulate(
    nb: Neuronenblitz,
    memory: EpisodicMemory,
    *,
    length: int = 5,
) -> List[float]:
    """Replay up to ``length`` episodes and return predicted rewards."""
    episodes: Iterable[EpisodicEntry] = memory.query({}, k=length)
    rewards = []
    for ep in episodes:
        out, _ = nb.dynamic_wander(ep.reward)
        rewards.append(out)
    return rewards


def register(*_: object) -> None:
    """No neuron or synapse types to register."""
    return
