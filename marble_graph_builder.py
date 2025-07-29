from __future__ import annotations

from typing import List, Iterable, Optional, Sequence

from marble_core import Core, Neuron, Synapse


def add_neuron_group(core: Core, count: int, activation: Optional[str] = None) -> List[int]:
    """Add ``count`` neurons to ``core``.

    Parameters
    ----------
    core : Core
        The marble core to extend.
    count : int
        Number of neurons to create.
    activation : Optional[str]
        Optional activation type to store in neuron metadata.

    Returns
    -------
    List[int]
        IDs of the created neurons.
    """
    ids: List[int] = []
    for _ in range(count):
        nid = len(core.neurons)
        neuron = Neuron(nid, value=0.0, tier="vram")
        if activation is not None:
            neuron.params["activation"] = activation
        core.neurons.append(neuron)
        ids.append(nid)
    return ids


def add_fully_connected_layer(
    core: Core,
    inputs: Sequence[int],
    out_dim: int,
    weights: Optional[Sequence[Sequence[float]]] = None,
    bias: Optional[Sequence[float]] = None,
    activation: Optional[str] = None,
) -> List[int]:
    """Create a fully connected layer.

    Parameters
    ----------
    core : Core
        The core to extend.
    inputs : Sequence[int]
        IDs of input neurons.
    out_dim : int
        Number of output neurons.
    weights : Optional[Sequence[Sequence[float]]]
        Weight matrix mapping ``inputs`` to outputs. If ``None``, weights default to zero.
    bias : Optional[Sequence[float]]
        Bias values for each output neuron.
    activation : Optional[str]
        Optional activation flag to assign to output neurons.

    Returns
    -------
    List[int]
        IDs of output neurons.
    """
    out_ids = add_neuron_group(core, out_dim, activation=activation)

    if weights is None:
        weights = [[0.0 for _ in inputs] for _ in range(out_dim)]

    for j, out_id in enumerate(out_ids):
        for i, in_id in enumerate(inputs):
            w = float(weights[j][i])
            syn = Synapse(in_id, out_id, weight=w)
            core.neurons[in_id].synapses.append(syn)
            core.synapses.append(syn)

    if bias is not None:
        bias_id = len(core.neurons)
        core.neurons.append(Neuron(bias_id, value=1.0, tier="vram"))
        for j, out_id in enumerate(out_ids):
            syn = Synapse(bias_id, out_id, weight=float(bias[j]))
            core.neurons[bias_id].synapses.append(syn)
            core.synapses.append(syn)

    return out_ids
