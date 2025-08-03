"""Memory management utilities for ``Neuronenblitz``."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import Neuronenblitz


def decay_memory_gates(nb: "Neuronenblitz") -> None:
    """Decay memory gate strengths over time."""
    for syn in list(nb.memory_gates.keys()):
        nb.memory_gates[syn] *= nb.memory_gate_decay
        if nb.memory_gates[syn] < 1e-6:
            del nb.memory_gates[syn]


def bias_with_episodic_memory(
    nb: "Neuronenblitz", entry, path, depth_remaining
):
    """Follow a stored episodic path to bias wandering.

    With probability ``nb.episodic_memory_prob`` a previously successful path
    stored in ``nb.episodic_memory`` is replayed for up to
    ``nb.episodic_sim_length`` steps.  Each traversed synapse applies the same
    side effects as during normal wandering and the remaining depth is
    decreased accordingly.

    Parameters
    ----------
    nb:
        ``Neuronenblitz`` instance.
    entry:
        Starting neuron for the wander.
    path:
        Current path as a list of ``(neuron, synapse)`` tuples where the first
        synapse is ``None``.
    depth_remaining:
        Number of wandering steps still allowed.

    Returns
    -------
    tuple
        ``(current_neuron, path, depth_remaining)`` reflecting the state after
        any episodic replay.
    """

    if nb.episodic_memory and random.random() < nb.episodic_memory_prob:
        mem_path = random.choice(list(nb.episodic_memory))
        steps = mem_path[: nb.episodic_sim_length]
        current = entry
        for syn in steps:
            if depth_remaining <= 0:
                break
            next_neuron = nb.core.neurons[syn.target]
            w = (
                syn.effective_weight(nb.last_context, nb.global_phase)
                if hasattr(syn, "effective_weight")
                else syn.weight
            )
            transmitted = nb.combine_fn(current.value, w)
            if hasattr(syn, "apply_side_effects"):
                syn.apply_side_effects(nb.core, current.value)
            if hasattr(syn, "update_echo"):
                syn.update_echo(current.value, nb.core.synapse_echo_decay)
            if nb.synaptic_fatigue_enabled and hasattr(syn, "update_fatigue"):
                syn.update_fatigue(nb.fatigue_increase, nb.fatigue_decay)
            syn.visit_count += 1
            if hasattr(next_neuron, "process"):
                next_neuron.value = next_neuron.process(transmitted)
            else:
                next_neuron.value = transmitted
            path.append((next_neuron, syn))
            current = next_neuron
            depth_remaining -= 1
        return current, path, depth_remaining
    return entry, path, depth_remaining
