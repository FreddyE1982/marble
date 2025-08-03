from __future__ import annotations

from typing import List, Iterable, Optional, Sequence

from marble_core import Core, Neuron, Synapse


def add_neuron_group(
    core: Core, count: int, activation: Optional[str] = None, activation_flag: bool = False
) -> List[int]:
    """Add ``count`` neurons to ``core``.

    Parameters
    ----------
    core : Core
        The marble core to extend.
    count : int
        Number of neurons to create.
    activation : Optional[str]
        Optional activation type to store in neuron metadata.
    activation_flag : bool
        Whether to mark neurons as active for message passing.

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
        neuron.params["activation_flag"] = activation_flag
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
    activation_flag: bool = False,
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
    activation_flag : bool
        Whether to mark output neurons as active.

    Returns
    -------
    List[int]
        IDs of output neurons.
    """
    out_ids = add_neuron_group(
        core, out_dim, activation=activation, activation_flag=activation_flag
    )

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


def linear_layer(
    core: Core,
    in_dim: int,
    out_dim: int,
    *,
    weights: Optional[Sequence[Sequence[float]]] = None,
    bias: Optional[Sequence[float]] = None,
    activation: Optional[str] = None,
    activation_flag: bool = False,
) -> tuple[List[int], List[int]]:
    """Create a fully connected layer with freshly allocated neurons.

    Parameters
    ----------
    core : Core
        Target core to extend.
    in_dim : int
        Number of input neurons to create.
    out_dim : int
        Number of output neurons to create.
    weights : Optional[Sequence[Sequence[float]]]
        Optional weight matrix. ``None`` initializes weights to zero.
    bias : Optional[Sequence[float]]
        Optional bias vector.
    activation : Optional[str]
        Activation type stored in output neuron metadata.

    Returns
    -------
    tuple[List[int], List[int]]
        IDs of created input and output neurons.
    """
    inputs = add_neuron_group(core, in_dim)
    outputs = add_fully_connected_layer(
        core,
        inputs,
        out_dim,
        weights=weights,
        bias=bias,
        activation=activation,
        activation_flag=activation_flag,
    )
    return inputs, outputs


def conv2d_layer(
    core: Core,
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    weights: Optional[Sequence[Sequence[Sequence[Sequence[float]]]]] = None,
    bias: Optional[Sequence[float]] = None,
) -> tuple[List[int], List[int]]:
    """Create a Conv2d-like layer using MARBLE primitives."""

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    inputs = add_neuron_group(core, in_channels)

    out_ids: List[int] = []
    if weights is None:
        weights = [
            [[ [0.0 for _ in range(kernel_size[1])] for _ in range(kernel_size[0])] for _ in range(in_channels)]
            for _ in range(out_channels)
        ]

    for j in range(out_channels):
        nid = len(core.neurons)
        neuron = Neuron(nid, value=0.0, tier="vram", neuron_type="conv2d")
        neuron.params["kernel"] = weights[j]
        neuron.params["stride"] = stride[0]
        neuron.params["padding"] = padding[0]
        core.neurons.append(neuron)

        for in_id in inputs:
            syn = Synapse(in_id, nid, weight=1.0)
            core.neurons[in_id].synapses.append(syn)
            core.synapses.append(syn)

        if bias is not None:
            bias_id = len(core.neurons)
            core.neurons.append(Neuron(bias_id, value=1.0, tier="vram"))
            b = float(bias[j])
            bsyn = Synapse(bias_id, nid, weight=b)
            core.neurons[bias_id].synapses.append(bsyn)
            core.synapses.append(bsyn)

        out_ids.append(nid)

    return inputs, out_ids
