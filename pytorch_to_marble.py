import argparse
import logging
import operator
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Tracer
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import (
    _AdaptiveAvgPoolNd,
    _AdaptiveMaxPoolNd,
    _AvgPoolNd,
    _MaxPoolNd,
)

from marble_core import Core, Neuron, Synapse
from marble_graph_builder import add_fully_connected_layer as gb_add_fc
from marble_utils import core_to_json, restore_hidden_states

logger = logging.getLogger(__name__)

# Standardised error message used for unsupported components
UNSUPPORTED_MSG = "{} is not supported for conversion"


class UnsupportedLayerError(Exception):
    """Raised when a layer type is not supported for conversion."""


class TracingFailedError(Exception):
    """Raised when ``torch.fx`` fails to trace a model."""


LayerConverter = Callable[..., List[int]]


LAYER_CONVERTERS: Dict[Type[torch.nn.Module], LayerConverter] = {}
FUNCTION_CONVERTERS: Dict[Callable, LayerConverter] = {}
METHOD_CONVERTERS: Dict[str, LayerConverter] = {}


@dataclass
class HiddenState:
    """Serializable description of an RNN hidden state tensor."""

    layer_index: int
    direction: str
    tensor: np.ndarray
    dtype: str
    device: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "layer_index": self.layer_index,
            "direction": self.direction,
            "tensor": self.tensor.tolist(),
            "shape": list(self.tensor.shape),
            "dtype": self.dtype,
            "device": self.device,
        }


def _extract_tensor(t: torch.Tensor) -> tuple[np.ndarray, str]:
    """Return tensor as CPU ``numpy`` array and record original device."""
    device = str(t.device)
    return t.detach().cpu().numpy(), device


class ConverterTracer(Tracer):
    """Tracer that treats registered converters as leaf modules."""

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if type(m) in LAYER_CONVERTERS:
            return True
        return super().is_leaf_module(m, module_qualified_name)


def register_converter(
    layer_type: Type[torch.nn.Module],
) -> Callable[[LayerConverter], LayerConverter]:
    """Decorator to register a layer conversion function."""

    def decorator(func: LayerConverter) -> LayerConverter:
        LAYER_CONVERTERS[layer_type] = func
        return func

    return decorator


def register_function_converter(
    func: Callable,
) -> Callable[[LayerConverter], LayerConverter]:
    """Decorator to register a converter for a functional op."""

    def decorator(conv: LayerConverter) -> LayerConverter:
        FUNCTION_CONVERTERS[func] = conv
        return conv

    return decorator


def register_method_converter(name: str) -> Callable[[LayerConverter], LayerConverter]:
    """Decorator to register a converter for Tensor methods."""

    def decorator(conv: LayerConverter) -> LayerConverter:
        METHOD_CONVERTERS[name] = conv
        return conv

    return decorator


def unsupported_layer(layer_type: Type[torch.nn.Module]) -> None:
    """Register a stub for an unsupported layer type.

    Parameters
    ----------
    layer_type:
        The PyTorch layer class to mark as unsupported.

    The registered stub raises :class:`UnsupportedLayerError` with a
    standardised message. This helper allows contributors to explicitly
    document missing conversions while providing clear guidance to users.
    """

    def _stub(*args, **kwargs) -> List[int]:  # pragma: no cover - simple error path
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer_type.__name__))

    LAYER_CONVERTERS[layer_type] = _stub

@register_function_converter(__import__("operator").getitem)
def _convert_getitem(
    func: Callable, core: Core, inputs: List[int], idx, *args, **kwargs
) -> List[int]:
    """Support tuple indexing by returning the input unchanged for idx 0."""
    if isinstance(idx, int) and idx == 0:
        return inputs
    raise UnsupportedLayerError(UNSUPPORTED_MSG.format("getitem"))


def _add_fully_connected_layer(
    core: Core, input_ids: List[int], layer: torch.nn.Linear
) -> List[int]:
    """Add a fully connected layer using the graph builder utility."""
    weights, w_device = _extract_tensor(layer.weight)
    bias, b_device = (None, None)
    if layer.bias is not None:
        bias, b_device = _extract_tensor(layer.bias)
    out_ids = gb_add_fc(core, input_ids, layer.out_features, weights=weights, bias=bias)
    for nid in out_ids:
        core.neurons[nid].params["weight_device"] = w_device
        if b_device is not None:
            core.neurons[nid].params["bias_device"] = b_device
    return out_ids


def _add_conv2d_layer(
    core: Core, input_ids: List[int], layer: torch.nn.Conv2d
) -> List[int]:
    """Add a Conv2d layer supporting multiple channels."""
    if layer.groups != 1:
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))
    if len(input_ids) != layer.in_channels:
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))

    out_ids: List[int] = []
    weight, w_device = _extract_tensor(layer.weight)
    bias, b_device = (None, None)
    if layer.bias is not None:
        bias, b_device = _extract_tensor(layer.bias)
    stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding

    for j in range(layer.out_channels):
        nid = len(core.neurons)
        neuron = Neuron(nid, value=0.0, tier="vram", neuron_type="conv2d")
        neuron.params["kernel"] = weight[j]
        neuron.params["weight_device"] = w_device
        neuron.params["stride"] = stride
        neuron.params["padding"] = padding
        core.neurons.append(neuron)

        for in_id in input_ids:
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
            neuron.params["bias_device"] = b_device

        out_ids.append(nid)

    return out_ids


def _add_pool2d_layer(
    core: Core,
    input_ids: List[int],
    layer: _MaxPoolNd | _AvgPoolNd,
    pool_type: str,
) -> List[int]:
    if len(input_ids) != 1:
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))
    inp = input_ids[0]
    nid = len(core.neurons)
    neuron = Neuron(nid, value=0.0, tier="vram", neuron_type=pool_type)
    ks = (
        layer.kernel_size[0]
        if isinstance(layer.kernel_size, tuple)
        else layer.kernel_size
    )
    stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    neuron.params["kernel_size"] = ks
    neuron.params["stride"] = stride
    neuron.params["padding"] = padding
    core.neurons.append(neuron)
    syn = Synapse(inp, nid, weight=1.0)
    core.neurons[inp].synapses.append(syn)
    core.synapses.append(syn)
    return [nid]


def _add_adaptive_pool2d_layer(
    core: Core,
    input_ids: List[int],
    layer: _AdaptiveAvgPoolNd | _AdaptiveMaxPoolNd,
    pool_type: str,
) -> List[int]:
    if len(input_ids) != 1:
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))
    inp = input_ids[0]
    nid = len(core.neurons)
    neuron = Neuron(nid, value=0.0, tier="vram", neuron_type=pool_type)
    neuron.params["output_size"] = (
        tuple(layer.output_size)
        if isinstance(layer.output_size, (list, tuple))
        else (layer.output_size,)
    )
    core.neurons.append(neuron)
    syn = Synapse(inp, nid, weight=1.0)
    core.neurons[inp].synapses.append(syn)
    core.synapses.append(syn)
    return [nid]


def _add_embedding_layer(
    core: Core, input_ids: List[int], layer: torch.nn.Embedding
) -> List[int]:
    """Add an Embedding or EmbeddingBag layer."""
    if len(input_ids) != 1:
        raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))
    inp = input_ids[0]
    nid = len(core.neurons)
    neuron = Neuron(nid, value=0.0, tier="vram", neuron_type="embedding")
    neuron.params["num_embeddings"] = int(layer.num_embeddings)
    neuron.params["embedding_dim"] = int(layer.embedding_dim)
    weights, w_device = _extract_tensor(layer.weight)
    neuron.params["weights"] = weights
    neuron.params["weight_device"] = w_device
    if layer.padding_idx is not None:
        neuron.params["padding_idx"] = int(layer.padding_idx)
    if layer.max_norm is not None:
        neuron.params["max_norm"] = float(layer.max_norm)
    core.neurons.append(neuron)
    syn = Synapse(inp, nid, weight=1.0)
    core.neurons[inp].synapses.append(syn)
    core.synapses.append(syn)
    return [nid]


def _add_recurrent_layer(
    core: Core,
    input_ids: List[int],
    layer: torch.nn.modules.rnn.RNNBase,
    neuron_type: str,
) -> List[int]:
    """Add a recurrent layer (RNN/LSTM/GRU)."""
    if getattr(layer, "proj_size", 0) != 0:
        raise UnsupportedLayerError(
            f"{layer.__class__.__name__} with projections is not supported"
        )

    num_directions = 2 if layer.bidirectional else 1
    prev_outputs = input_ids
    out_ids: List[int] = []

    for layer_idx in range(layer.num_layers):
        layer_out: List[int] = []
        for d in range(num_directions):
            suffix = "" if d == 0 else "_reverse"
            weight_ih, wih_device = _extract_tensor(
                getattr(layer, f"weight_ih_l{layer_idx}{suffix}")
            )
            weight_hh, whh_device = _extract_tensor(
                getattr(layer, f"weight_hh_l{layer_idx}{suffix}")
            )
            bias = None
            b_device = None
            if layer.bias:
                bias_ih, bih_device = _extract_tensor(
                    getattr(layer, f"bias_ih_l{layer_idx}{suffix}")
                )
                bias_hh, bhh_device = _extract_tensor(
                    getattr(layer, f"bias_hh_l{layer_idx}{suffix}")
                )
                bias = bias_ih + bias_hh
                b_device = bih_device

            state_tensor = np.zeros(layer.hidden_size, dtype=np.float32)
            core.params.setdefault("hidden_states", [])
            core.params.setdefault("hidden_state_version", 1)
            core.params["hidden_states"].append(
                asdict(
                    HiddenState(
                        layer_index=layer_idx,
                        direction="forward" if d == 0 else "reverse",
                        tensor=state_tensor,
                        dtype="float32",
                        device=wih_device,
                    )
                )
            )

            ids: List[int] = []
            for j in range(layer.hidden_size):
                nid = len(core.neurons)
                neuron = Neuron(nid, value=0.0, tier="vram", neuron_type=neuron_type)
                neuron.params["weight_ih"] = weight_ih[j]
                neuron.params["weight_ih_device"] = wih_device
                neuron.params["weight_hh"] = weight_hh[j]
                neuron.params["weight_hh_device"] = whh_device
                if bias is not None:
                    neuron.params["bias"] = float(bias[j])
                    neuron.params["bias_device"] = b_device
                if neuron_type == "rnn":
                    neuron.params["nonlinearity"] = layer.nonlinearity
                neuron.params["input_size"] = int(
                    layer.input_size
                    if layer_idx == 0
                    else layer.hidden_size * num_directions
                )
                neuron.params["hidden_size"] = int(layer.hidden_size)
                neuron.params["layer_index"] = layer_idx
                neuron.params["direction"] = "forward" if d == 0 else "reverse"
                core.neurons.append(neuron)
                ids.append(nid)
            layer_out.extend(ids)

            for out_id in ids:
                for in_id in prev_outputs:
                    syn = Synapse(in_id, out_id, weight=1.0)
                    core.neurons[in_id].synapses.append(syn)
                    core.synapses.append(syn)
                for h_id in ids:
                    syn = Synapse(h_id, out_id, weight=1.0)
                    core.neurons[h_id].synapses.append(syn)
                    core.synapses.append(syn)

        prev_outputs = layer_out
        out_ids = layer_out

    return out_ids


def _elementwise_binary_op(
    core: Core,
    left: List[int],
    right: List[int],
    op_type: str,
) -> List[int]:
    """Create neurons performing an element-wise binary operation."""
    if len(left) != len(right):
        raise UnsupportedLayerError(
            f"{op_type} requires tensors of equal length for conversion"
        )
    out: List[int] = []
    for a, b in zip(left, right):
        nid = len(core.neurons)
        neuron = Neuron(nid, value=0.0, tier="vram", neuron_type="linear")
        neuron.params["op"] = op_type
        core.neurons.append(neuron)
        for src in (a, b):
            syn = Synapse(src, nid, weight=1.0)
            core.neurons[src].synapses.append(syn)
            core.synapses.append(syn)
        out.append(nid)
    return out


@register_converter(torch.nn.Linear)
def _convert_linear(
    layer: torch.nn.Linear, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_fully_connected_layer(core, inputs, layer)


@register_converter(torch.nn.Conv2d)
def _convert_conv2d(
    layer: torch.nn.Conv2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_conv2d_layer(core, inputs, layer)


@register_converter(torch.nn.ReLU)
def _convert_relu(
    layer: torch.nn.ReLU, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        core.neurons[nid].params["activation"] = "relu"
    return inputs


@register_converter(torch.nn.Sigmoid)
def _convert_sigmoid(
    layer: torch.nn.Sigmoid, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "sigmoid"
    return inputs


@register_converter(torch.nn.Tanh)
def _convert_tanh(
    layer: torch.nn.Tanh, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "tanh"
    return inputs


@register_converter(torch.nn.GELU)
def _convert_gelu(
    layer: torch.nn.GELU, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "gelu"
    return inputs


@register_converter(torch.nn.Softmax)
def _convert_softmax(
    layer: torch.nn.Softmax, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    """Convert a ``Softmax`` layer."""
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "softmax"
        n.params["dim"] = int(layer.dim)
    return inputs


@register_function_converter(F.relu)
@register_method_converter("relu")
def _convert_f_relu(
    func: Callable, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _convert_relu(torch.nn.ReLU(), core, inputs)


@register_function_converter(F.sigmoid)
@register_function_converter(torch.sigmoid)
@register_method_converter("sigmoid")
def _convert_f_sigmoid(
    func: Callable, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _convert_sigmoid(torch.nn.Sigmoid(), core, inputs)


@register_function_converter(F.tanh)
@register_function_converter(torch.tanh)
@register_method_converter("tanh")
def _convert_f_tanh(
    func: Callable, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _convert_tanh(torch.nn.Tanh(), core, inputs)


@register_function_converter(F.softmax)
@register_function_converter(torch.softmax)
@register_method_converter("softmax")
def _convert_f_softmax(
    func: Callable, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    dim = kwargs.get("dim", args[0] if args else -1)
    return _convert_softmax(torch.nn.Softmax(dim=dim), core, inputs)


@register_function_converter(torch.add)
@register_function_converter(operator.add)
@register_method_converter("add")
def _convert_add(
    func: Callable, core: Core, inputs: List[int], other, *args, **kwargs
) -> List[int]:
    if not isinstance(other, list):
        raise UnsupportedLayerError("add expects tensor input")
    return _elementwise_binary_op(core, inputs, other, "add")


@register_function_converter(torch.mul)
@register_function_converter(operator.mul)
@register_method_converter("mul")
def _convert_mul(
    func: Callable, core: Core, inputs: List[int], other, *args, **kwargs
) -> List[int]:
    if not isinstance(other, list):
        raise UnsupportedLayerError("mul expects tensor input")
    return _elementwise_binary_op(core, inputs, other, "mul")


@register_converter(torch.nn.Dropout)
def _convert_dropout(
    layer: torch.nn.Dropout, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "dropout"
        n.params["p"] = float(layer.p)
    return inputs


@register_converter(torch.nn.MaxPool2d)
def _convert_maxpool2d(
    layer: torch.nn.MaxPool2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_pool2d_layer(core, inputs, layer, "maxpool2d")


@register_converter(torch.nn.AvgPool2d)
def _convert_avgpool2d(
    layer: torch.nn.AvgPool2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_pool2d_layer(core, inputs, layer, "avgpool2d")


@register_converter(torch.nn.AdaptiveAvgPool2d)
def _convert_adaptiveavgpool2d(
    layer: torch.nn.AdaptiveAvgPool2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_adaptive_pool2d_layer(core, inputs, layer, "avgpool2d")


@register_converter(torch.nn.AdaptiveMaxPool2d)
def _convert_adaptivemaxpool2d(
    layer: torch.nn.AdaptiveMaxPool2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_adaptive_pool2d_layer(core, inputs, layer, "maxpool2d")


@register_converter(torch.nn.Embedding)
def _convert_embedding(
    layer: torch.nn.Embedding, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_embedding_layer(core, inputs, layer)


@register_converter(torch.nn.EmbeddingBag)
def _convert_embeddingbag(
    layer: torch.nn.EmbeddingBag, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_embedding_layer(core, inputs, layer)


@register_converter(torch.nn.RNN)
def _convert_rnn(
    layer: torch.nn.RNN, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_recurrent_layer(core, inputs, layer, "rnn")


@register_converter(torch.nn.LSTM)
def _convert_lstm(
    layer: torch.nn.LSTM, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_recurrent_layer(core, inputs, layer, "lstm")


@register_converter(torch.nn.GRU)
def _convert_gru(
    layer: torch.nn.GRU, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_recurrent_layer(core, inputs, layer, "gru")


class GlobalAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def __init__(self) -> None:
        super().__init__((1, 1))


@register_converter(GlobalAvgPool2d)
def _convert_globalavgpool2d(
    layer: GlobalAvgPool2d, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    return _add_adaptive_pool2d_layer(core, inputs, layer, "avgpool2d")


@register_converter(torch.nn.BatchNorm1d)
@register_converter(torch.nn.BatchNorm2d)
def _convert_batchnorm(
    layer: _BatchNorm, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "batchnorm"
        n.params["momentum"] = float(layer.momentum or 0.1)
        n.params["eps"] = float(layer.eps)
    return inputs


@register_converter(torch.nn.LayerNorm)
def _convert_layernorm(
    layer: torch.nn.LayerNorm, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    """Convert a ``LayerNorm`` layer."""
    normalized_shape = (
        tuple(layer.normalized_shape)
        if isinstance(layer.normalized_shape, (list, tuple))
        else (int(layer.normalized_shape),)
    )
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "layernorm"
        n.params["normalized_shape"] = normalized_shape
        n.params["eps"] = float(layer.eps)
    return inputs


@register_converter(torch.nn.GroupNorm)
def _convert_groupnorm(
    layer: torch.nn.GroupNorm, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    """Convert a ``GroupNorm`` layer."""
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "groupnorm"
        n.params["num_groups"] = int(layer.num_groups)
        n.params["eps"] = float(layer.eps)
    return inputs


@register_converter(torch.nn.Flatten)
def _convert_flatten(
    layer: torch.nn.Flatten, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "flatten"
    return inputs


@register_converter(torch.nn.Unflatten)
def _convert_unflatten(
    layer: torch.nn.Unflatten, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "unflatten"
        n.params["dim"] = int(layer.dim)
        n.params["unflattened_size"] = tuple(layer.unflattened_size)
    return inputs


@register_converter(torch.nn.Sequential)
def _convert_sequential(
    layer: torch.nn.Sequential, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    """Recursively convert all submodules of a ``Sequential`` container."""
    out = inputs
    for sub in layer:
        converter = _get_converter(sub)
        out = converter(sub, core, out)
    return out


@register_converter(torch.nn.ModuleList)
def _convert_modulelist(
    layer: torch.nn.ModuleList, core: Core, inputs: List[int], *args, **kwargs
) -> List[int]:
    """Iterate over a ``ModuleList`` converting each module in order."""
    out = inputs
    for sub in layer:
        converter = _get_converter(sub)
        out = converter(sub, core, out)
    return out


@register_function_converter(torch.reshape)
def _convert_reshape(
    func: Callable, core: Core, inputs: List[int], *shape, **kwargs
) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "reshape"
        if shape:
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                n.params["shape"] = tuple(shape[0])
            else:
                n.params["shape"] = tuple(shape)
    return inputs


@register_method_converter("view")
def _convert_view(
    name: str, core: Core, inputs: List[int], *shape, **kwargs
) -> List[int]:
    return _convert_reshape(torch.reshape, core, inputs, *shape, **kwargs)


def _get_converter(layer: torch.nn.Module) -> LayerConverter:
    for cls in LAYER_CONVERTERS:
        if isinstance(layer, cls):
            return LAYER_CONVERTERS[cls]
    raise UnsupportedLayerError(UNSUPPORTED_MSG.format(layer.__class__.__name__))


def _get_function_converter(func: Callable) -> LayerConverter:
    if func in FUNCTION_CONVERTERS:
        return FUNCTION_CONVERTERS[func]
    raise UnsupportedLayerError(UNSUPPORTED_MSG.format(func.__name__))


def _get_method_converter(name: str) -> LayerConverter:
    if name in METHOD_CONVERTERS:
        return METHOD_CONVERTERS[name]
    raise UnsupportedLayerError(UNSUPPORTED_MSG.format(name))


def _print_dry_run_summary(
    core: Core, node_outputs: Dict[str, List[int]], layer_synapses: Dict[str, int]
) -> None:
    """Print summary statistics for a dry-run conversion."""
    print(
        f"[DRY RUN] created {len(core.neurons)} neurons and {len(core.synapses)} synapses"
    )
    for name, ids in node_outputs.items():
        if name != "output":
            syns = layer_synapses.get(name, 0)
            print(f"[DRY RUN] {name}: {len(ids)} neurons, {syns} synapses")


def convert_model(
    model: torch.nn.Module,
    core_params: Dict | None = None,
    dry_run: bool = False,
    return_summary: bool = False,
    restore_hidden: bool = False,
) -> Core | tuple[Core, Dict[str, Dict]]:
    """Convert ``model`` into a MARBLE ``Core``.

    Parameters
    ----------
    model:
        PyTorch module to convert.
    core_params:
        Optional parameters used to initialise the :class:`Core`.
    dry_run:
        When ``True`` the core is constructed but not serialised.
    return_summary:
        If set, layer statistics are returned alongside the core.
    restore_hidden:
        Restore any serialised RNN hidden states into the resulting core.
    """
    if core_params is None:
        core_params = {
            "xmin": -2.0,
            "xmax": 1.0,
            "ymin": -1.5,
            "ymax": 1.5,
            "width": 1,
            "height": 1,
            "max_iter": 1,
            "vram_limit_mb": 0.1,
            "ram_limit_mb": 0.1,
            "disk_limit_mb": 0.1,
        }
    try:
        tracer = ConverterTracer()
        graph = tracer.trace(model)
        traced = GraphModule(model, graph)
    except Exception as exc:
        raise TracingFailedError(f"FX tracing failed: {exc}") from exc
    logger.info("Tracing succeeded with %d nodes", len(traced.graph.nodes))
    core = Core(core_params, formula="0", formula_num_neurons=0)
    node_outputs: Dict[str, List[int]] = {}
    layer_synapses: Dict[str, int] = {}
    for node in traced.graph.nodes:
        if node.op == "placeholder":
            input_tensor = node.meta.get("tensor_meta")
            if input_tensor is not None:
                in_dim = input_tensor.shape[1]
            else:
                if hasattr(model, "input_size"):
                    size = model.input_size
                    in_dim = size[0] if isinstance(size, tuple) else size
                else:
                    in_dim = 1
            ids = []
            logger.info("Creating %d input neurons for %s", in_dim, node.name)
            for _ in range(in_dim):
                nid = len(core.neurons)
                core.neurons.append(Neuron(nid, value=0.0, tier="vram"))
                ids.append(nid)
            node_outputs[node.name] = ids
        elif node.op == "call_module":
            layer = dict(model.named_modules())[node.target]
            logger.info(
                "Converting layer %s (%s)", node.target, layer.__class__.__name__
            )
            converter = _get_converter(layer)
            inp = node_outputs[node.args[0].name]
            pre_syn = len(core.synapses)
            out = converter(layer, core, inp)
            layer_synapses[node.name] = len(core.synapses) - pre_syn
            node_outputs[node.name] = out
        elif node.op == "call_function":
            logger.info(
                "Converting function %s",
                getattr(node.target, "__name__", str(node.target)),
            )
            converter = _get_function_converter(node.target)
            resolved_args: List = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    resolved_args.append(node_outputs[arg.name])
                else:
                    resolved_args.append(arg)
            inp = resolved_args[0]
            extra_args = resolved_args[1:]
            pre_syn = len(core.synapses)
            out = converter(node.target, core, inp, *extra_args, **node.kwargs)
            layer_synapses[node.name] = len(core.synapses) - pre_syn
            node_outputs[node.name] = out
        elif node.op == "get_attr":
            # Attributes such as parameters are accessed directly by subsequent modules.
            pass
        elif node.op == "call_method":
            logger.info("Converting method %s", node.target)
            converter = _get_method_converter(node.target)
            resolved_args: List = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    resolved_args.append(node_outputs[arg.name])
                else:
                    resolved_args.append(arg)
            inp = resolved_args[0]
            extra_args = resolved_args[1:]
            pre_syn = len(core.synapses)
            out = converter(node.target, core, inp, *extra_args, **node.kwargs)
            layer_synapses[node.name] = len(core.synapses) - pre_syn
            node_outputs[node.name] = out
        elif node.op == "output":
            pass
        else:
            raise UnsupportedLayerError(
                f"Operation {node.op} is not supported for conversion"
            )
    summary = {
        "neurons": len(core.neurons),
        "synapses": len(core.synapses),
        "layers": {
            name: {
                "neurons": len(ids),
                "synapses": layer_synapses.get(name, 0),
            }
            for name, ids in node_outputs.items()
            if name != "output"
        },
    }
    if dry_run or return_summary:
        _print_dry_run_summary(core, node_outputs, layer_synapses)
    if restore_hidden:
        restore_hidden_states(core)
    if return_summary:
        return core, summary
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch model to MARBLE JSON")
    parser.add_argument("--pytorch", required=True, help="Path to PyTorch model")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--restore-hidden",
        action="store_true",
        help="Restore serialised RNN hidden states into the converted core",
    )
    args = parser.parse_args()

    from torch_model_io import load_model_auto

    model = load_model_auto(args.pytorch)
    core = convert_model(model, restore_hidden=args.restore_hidden)
    js = core_to_json(core)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(js)


if __name__ == "__main__":
    main()
