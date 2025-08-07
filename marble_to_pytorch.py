"""Utilities to convert a MARBLE Core graph back into a PyTorch ``nn.Module``.

The converter walks the neuron/synapse structure stored in :class:`marble_core.Core`
and reconstructs a sequential ``nn.Module`` using layer metadata.  Only layers
that are supported by ``pytorch_to_marble`` are reconstructed: fully-connected
layers with optional dropout and activation functions (ReLU, Sigmoid, Tanh,
GELU).  Additional layer types can be added iteratively.

The conversion algorithm assumes that the ``Core`` was produced by
``pytorch_to_marble.convert_model`` which organises neurons in contiguous blocks
per layer and inserts a dedicated bias neuron with value ``1.0`` after each
layer when biases are present.  Neurons contain metadata about activations and
in-place operations in ``Neuron.params`` and ``Neuron.neuron_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

import torch
from torch import nn

from marble_core import Core, Neuron, Synapse


@dataclass
class _Layer:
    """Internal representation of a linear layer extracted from the Core."""

    inputs: List[int]
    outputs: List[int]
    weight: torch.Tensor
    bias: torch.Tensor | None
    device: torch.device


def _build_incoming(core: Core) -> Dict[int, List[Synapse]]:
    """Map target neuron id to incoming synapses."""
    incoming: Dict[int, List[Synapse]] = {n.id: [] for n in core.neurons}
    for syn in core.synapses:
        incoming[syn.target].append(syn)
    return incoming


def _find_input_neurons(core: Core, incoming: Dict[int, List[Synapse]]) -> List[int]:
    """Return ids of neurons with no incoming synapses (excluding bias neurons)."""
    return [n.id for n in core.neurons if n.value != 1.0 and not incoming[n.id]]


def _bias_neurons(core: Core, incoming: Dict[int, List[Synapse]]) -> Set[int]:
    return {n.id for n in core.neurons if n.value == 1.0 and not incoming[n.id]}


def _extract_linear_layer(
    core: Core,
    current: List[int],
    incoming: Dict[int, List[Synapse]],
    processed: Set[int],
    bias_ids: Set[int],
) -> _Layer | None:
    """Find the next linear layer whose sources come from ``current``.

    Returns ``None`` when no further layer can be extracted."""
    next_ids: List[int] = []
    allowed = set(current) | bias_ids
    for n in core.neurons:
        if n.id in processed or n.id in bias_ids:
            continue
        sources = {s.source for s in incoming[n.id]}
        if sources and sources.issubset(allowed):
            next_ids.append(n.id)
    if not next_ids:
        return None
    next_ids.sort()

    weight = torch.zeros(len(next_ids), len(current), dtype=torch.float32)
    bias = torch.zeros(len(next_ids), dtype=torch.float32)
    has_bias = False
    device = torch.device(
        core.neurons[next_ids[0]].params.get("weight_device", "cpu")
    )

    for j, nid in enumerate(next_ids):
        for syn in incoming[nid]:
            if syn.source in current:
                idx = current.index(syn.source)
                weight[j, idx] = float(syn.weight)
            elif syn.source in bias_ids:
                bias[j] = float(syn.weight)
                has_bias = True
                processed.add(syn.source)
    return _Layer(
        inputs=current,
        outputs=next_ids,
        weight=weight.to(device),
        bias=bias.to(device) if has_bias else None,
        device=device,
    )


def _apply_inplace_ops(modules: List[nn.Module], neurons: Sequence[Neuron]) -> None:
    """Append PyTorch modules for in-place ops encoded in ``neurons``."""
    if neurons and all(n.neuron_type == "dropout" for n in neurons):
        p = float(neurons[0].params.get("p", 0.5))
        modules.append(nn.Dropout(p))
        for n in neurons:
            n.neuron_type = "standard"
    act = neurons[0].params.get("activation") if neurons else None
    if act == "relu":
        modules.append(nn.ReLU())
        for n in neurons:
            n.params.pop("activation", None)
    elif act == "gelu":
        modules.append(nn.GELU())
        for n in neurons:
            n.params.pop("activation", None)
    else:
        nt = neurons[0].neuron_type if neurons else None
        if nt == "sigmoid":
            modules.append(nn.Sigmoid())
            for n in neurons:
                n.neuron_type = "standard"
        elif nt == "tanh":
            modules.append(nn.Tanh())
            for n in neurons:
                n.neuron_type = "standard"


def convert_core(core: Core) -> nn.Module:
    """Convert a :class:`Core` into a PyTorch ``nn.Module``.

    Parameters
    ----------
    core:
        The MARBLE ``Core`` produced by ``pytorch_to_marble``.

    Returns
    -------
    nn.Module
        A sequential PyTorch module replicating the original model's behaviour.
    """
    incoming = _build_incoming(core)
    inputs = _find_input_neurons(core, incoming)
    bias_ids = _bias_neurons(core, incoming)
    processed: Set[int] = set(inputs) | bias_ids

    modules: List[nn.Module] = []
    current = inputs
    while True:
        _apply_inplace_ops(modules, [core.neurons[i] for i in current])
        layer = _extract_linear_layer(core, current, incoming, processed, bias_ids)
        if layer is None:
            break
        linear = nn.Linear(len(layer.inputs), len(layer.outputs), bias=layer.bias is not None)
        linear.weight.data = layer.weight
        if layer.bias is not None:
            linear.bias.data = layer.bias
        modules.append(linear.to(layer.device))
        processed.update(layer.outputs)
        current = layer.outputs

    class MarbleModule(nn.Module):
        def __init__(self, modules: List[nn.Module]):
            super().__init__()
            self.net = nn.Sequential(*modules)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return MarbleModule(modules)
