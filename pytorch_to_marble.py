import argparse
from typing import Callable, Dict, List, Type

import torch
from torch.fx import symbolic_trace
from torch.nn.modules.batchnorm import _BatchNorm

from marble_core import Core, Neuron, Synapse
from marble_utils import core_to_json


class UnsupportedLayerError(Exception):
    """Raised when a layer type is not supported for conversion."""


LayerConverter = Callable[[torch.nn.Module, Core, List[int]], List[int]]


LAYER_CONVERTERS: Dict[Type[torch.nn.Module], LayerConverter] = {}


def register_converter(
    layer_type: Type[torch.nn.Module],
) -> Callable[[LayerConverter], LayerConverter]:
    """Decorator to register a layer conversion function."""

    def decorator(func: LayerConverter) -> LayerConverter:
        LAYER_CONVERTERS[layer_type] = func
        return func

    return decorator


def _add_fully_connected_layer(
    core: Core, input_ids: List[int], layer: torch.nn.Linear
) -> List[int]:
    out_ids = []
    for _ in range(layer.out_features):
        nid = len(core.neurons)
        core.neurons.append(Neuron(nid, value=0.0, tier="vram"))
        out_ids.append(nid)
    weight = layer.weight.detach().cpu().numpy()
    for j, out_id in enumerate(out_ids):
        for i, in_id in enumerate(input_ids):
            w = float(weight[j, i])
            syn = Synapse(in_id, out_id, weight=w)
            core.neurons[in_id].synapses.append(syn)
            core.synapses.append(syn)
    if layer.bias is not None:
        bias_id = len(core.neurons)
        core.neurons.append(Neuron(bias_id, value=1.0, tier="vram"))
        bias = layer.bias.detach().cpu().numpy()
        for j, out_id in enumerate(out_ids):
            syn = Synapse(bias_id, out_id, weight=float(bias[j]))
            core.neurons[bias_id].synapses.append(syn)
            core.synapses.append(syn)
    return out_ids


def _add_conv2d_layer(
    core: Core, input_ids: List[int], layer: torch.nn.Conv2d
) -> List[int]:
    if layer.in_channels != 1 or len(input_ids) != 1 or layer.groups != 1:
        raise UnsupportedLayerError("Conv2d with in_channels!=1 is not supported for conversion")
    out_ids = []
    weight = layer.weight.detach().cpu().numpy()
    stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    inp = input_ids[0]
    for j in range(layer.out_channels):
        nid = len(core.neurons)
        neuron = Neuron(nid, value=0.0, tier="vram", neuron_type="conv2d")
        neuron.params["kernel"] = weight[j, 0]
        neuron.params["stride"] = stride
        neuron.params["padding"] = padding
        core.neurons.append(neuron)
        syn = Synapse(inp, nid, weight=1.0)
        core.neurons[inp].synapses.append(syn)
        core.synapses.append(syn)
        if layer.bias is not None:
            bias_id = len(core.neurons)
            core.neurons.append(Neuron(bias_id, value=1.0, tier="vram"))
            b = float(layer.bias.detach().cpu().numpy()[j])
            bsyn = Synapse(bias_id, nid, weight=b)
            core.neurons[bias_id].synapses.append(bsyn)
            core.synapses.append(bsyn)
        out_ids.append(nid)
    return out_ids


@register_converter(torch.nn.Linear)
def _convert_linear(layer: torch.nn.Linear, core: Core, inputs: List[int]) -> List[int]:
    return _add_fully_connected_layer(core, inputs, layer)


@register_converter(torch.nn.Conv2d)
def _convert_conv2d(layer: torch.nn.Conv2d, core: Core, inputs: List[int]) -> List[int]:
    return _add_conv2d_layer(core, inputs, layer)


@register_converter(torch.nn.ReLU)
def _convert_relu(layer: torch.nn.ReLU, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        core.neurons[nid].params["activation"] = "relu"
    return inputs


@register_converter(torch.nn.Sigmoid)
def _convert_sigmoid(layer: torch.nn.Sigmoid, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "sigmoid"
    return inputs


@register_converter(torch.nn.Tanh)
def _convert_tanh(layer: torch.nn.Tanh, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "tanh"
    return inputs


@register_converter(torch.nn.GELU)
def _convert_gelu(layer: torch.nn.GELU, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "gelu"
    return inputs


@register_converter(torch.nn.Dropout)
def _convert_dropout(layer: torch.nn.Dropout, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "dropout"
        n.params["p"] = float(layer.p)
    return inputs


@register_converter(torch.nn.BatchNorm1d)
@register_converter(torch.nn.BatchNorm2d)
def _convert_batchnorm(layer: _BatchNorm, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        n = core.neurons[nid]
        n.neuron_type = "batchnorm"
        n.params["momentum"] = float(layer.momentum or 0.1)
        n.params["eps"] = float(layer.eps)
    return inputs


@register_converter(torch.nn.Flatten)
def _convert_flatten(layer: torch.nn.Flatten, core: Core, inputs: List[int]) -> List[int]:
    for nid in inputs:
        core.neurons[nid].neuron_type = "flatten"
    return inputs


def _get_converter(layer: torch.nn.Module) -> LayerConverter:
    for cls in LAYER_CONVERTERS:
        if isinstance(layer, cls):
            return LAYER_CONVERTERS[cls]
    raise UnsupportedLayerError(
        f"{layer.__class__.__name__} is not supported for conversion"
    )


def convert_model(
    model: torch.nn.Module, core_params: Dict | None = None, dry_run: bool = False
) -> Core:
    """Convert ``model`` into a MARBLE ``Core``."""
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
    traced = symbolic_trace(model)
    core = Core(core_params, formula="0", formula_num_neurons=0)
    node_outputs: Dict[str, List[int]] = {}
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
            for _ in range(in_dim):
                nid = len(core.neurons)
                core.neurons.append(Neuron(nid, value=0.0, tier="vram"))
                ids.append(nid)
            node_outputs[node.name] = ids
        elif node.op == "call_module":
            layer = dict(model.named_modules())[node.target]
            converter = _get_converter(layer)
            inp = node_outputs[node.args[0].name]
            out = converter(layer, core, inp)
            node_outputs[node.name] = out
        elif node.op == "output":
            pass
        else:
            raise UnsupportedLayerError(
                f"Operation {node.op} is not supported for conversion"
            )
    if dry_run:
        return core
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch model to MARBLE JSON")
    parser.add_argument("--pytorch", required=True, help="Path to PyTorch model")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    model = torch.load(args.pytorch, map_location="cpu")
    core = convert_model(model)
    js = core_to_json(core)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(js)


if __name__ == "__main__":
    main()
