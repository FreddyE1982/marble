"""Demonstrate runtime graph changes for message passing.

This example builds a tiny MARBLE core with two neurons and shows how
adding and removing synapses during execution influences message
propagation. The operations work on both CPU and GPU depending on the
available backend.
"""

from __future__ import annotations

import os, sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensor_backend as tb
from core.message_passing import perform_message_passing
from marble_core import Core, Synapse
from marble_graph_builder import add_neuron_group


def _add_synapse(core: Core, src: int, dst: int, weight: float) -> Synapse:
    """Create a synapse linking ``src`` -> ``dst`` and register it."""

    syn = Synapse(src, dst, weight)
    core.neurons[src].synapses.append(syn)
    core.synapses.append(syn)
    return syn


def run_demo(return_history: bool = False) -> Tuple[List[List[float]], List[int]]:
    """Run dynamic message passing demo.

    Parameters
    ----------
    return_history : bool, optional
        If ``True`` return representation snapshots and synapse counts instead
        of printing them.

    Returns
    -------
    Tuple[List[List[float]], List[int]]
        History of neuron representations (after each stage) and synapse counts
        corresponding to those stages.
    """

    xp = tb.xp()
    core = Core(params={"representation_size": 2})
    # start with an empty graph
    core.neurons = []
    core.synapses = []

    # Create two neurons marked active for message passing.
    n0, n1 = add_neuron_group(core, 2, activation_flag=True)
    core.neurons[n0].representation = xp.array([1.0, 0.0], dtype=xp.float32)
    core.neurons[n1].representation = xp.array([0.0, 1.0], dtype=xp.float32)

    # Initial connection n0 -> n1
    syn01 = _add_synapse(core, n0, n1, 1.0)
    perform_message_passing(core)
    reps1 = [core.neurons[n].representation.copy() for n in (n0, n1)]
    syn_counts = [len(core.synapses)]

    # Dynamically add reverse edge n1 -> n0
    _add_synapse(core, n1, n0, 0.5)
    perform_message_passing(core)
    reps2 = [core.neurons[n].representation.copy() for n in (n0, n1)]
    syn_counts.append(len(core.synapses))

    # Remove the original edge
    core.neurons[n0].synapses.remove(syn01)
    core.synapses.remove(syn01)
    perform_message_passing(core)
    reps3 = [core.neurons[n].representation.copy() for n in (n0, n1)]
    syn_counts.append(len(core.synapses))

    if return_history:
        history = [
            [tb.to_numpy(rep).tolist() for rep in reps]
            for reps in (reps1, reps2, reps3)
        ]
        return history, syn_counts

    print("After initial pass:", reps1)
    print("After adding synapse:", reps2)
    print("After removal:", reps3)
    print("Synapse counts:", syn_counts)
    return [], syn_counts


if __name__ == "__main__":
    run_demo()
