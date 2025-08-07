import numpy as np
import torch

import marble_core
from marble_core import _B1, _B2, _W1, _W2, Core, configure_representation_size


class LinearLayer(torch.nn.Module):
    """Simple linear layer using explicit weight and bias parameters."""

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor, device: torch.device
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.to(device))
        self.bias = torch.nn.Parameter(bias.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return x @ self.weight + self.bias


class GraphLayer(torch.nn.Module):
    """Layer performing message passing based on a synapse adjacency matrix."""

    def __init__(self, adjacency: np.ndarray, device: torch.device) -> None:
        super().__init__()
        weight = torch.tensor(adjacency, dtype=torch.float32, device=device)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        return x @ self.weight


class MarbleTorchAdapter(torch.nn.Module):
    """PyTorch module mirroring Marble's message passing MLP."""

    def __init__(self, core: Core, device: torch.device | None = None) -> None:
        super().__init__()
        self.core = core
        device = device or torch.device("cpu")
        layer_params = core.params.get("mlp_layers")
        if layer_params is None:
            layer_params = [
                {"weight": _W1, "bias": _B1},
                {"weight": _W2, "bias": _B2},
            ]
        self.layers = torch.nn.ModuleList()
        for lp in layer_params:
            w = torch.tensor(lp["weight"], dtype=torch.float32)
            b = torch.tensor(lp["bias"], dtype=torch.float32)
            self.layers.append(LinearLayer(w, b, device))

        # Convenience handles for the classic 2-layer case
        if len(self.layers) > 0:
            self.w1 = self.layers[0].weight
            self.b1 = self.layers[0].bias
        if len(self.layers) > 1:
            self.w2 = self.layers[1].weight
            self.b2 = self.layers[1].bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h


def core_to_torch(core: Core) -> MarbleTorchAdapter:
    """Return a ``MarbleTorchAdapter`` wrapping ``core`` for inference."""
    return MarbleTorchAdapter(core)


def torch_to_core(model: torch.nn.Module, core: Core) -> None:
    """Update Marble's MLP weights from a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Module containing linear layers with ``weight`` and ``bias``.
    core : Core
        Core instance whose representation size will be adjusted to match
        ``model``.
    """
    first_weight = next(model.parameters()).detach()
    rep_size = first_weight.shape[0]
    configure_representation_size(rep_size)
    layers = []
    for layer in getattr(model, "layers", []):
        w = layer.weight.detach().cpu().numpy().astype(np.float64)
        b = layer.bias.detach().cpu().numpy().astype(np.float64)
        layers.append({"weight": np.round(w, 8), "bias": np.round(b, 8)})
    if layers:
        core.params["mlp_layers"] = layers
        if len(layers) > 0:
            marble_core._W1 = layers[0]["weight"]
            marble_core._B1 = layers[0]["bias"]
        if len(layers) > 1:
            marble_core._W2 = layers[1]["weight"]
            marble_core._B2 = layers[1]["bias"]
    for n in core.neurons:
        if n.representation.shape != (rep_size,):
            n.representation = np.zeros(rep_size, dtype=float)


def core_to_torch_graph(
    core: Core, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edge indices and node features as torch tensors."""
    edge_index = [[s.source, s.target] for s in core.synapses]
    edge_tensor = (
        torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long, device=device)
    )
    feats = torch.tensor(
        [n.representation for n in core.neurons], dtype=torch.float32, device=device
    )
    return edge_tensor, feats


def torch_graph_to_core(
    edge_index: torch.Tensor, features: torch.Tensor, params: dict
) -> Core:
    """Create a Core from edge indices and node features."""
    core = Core(params)
    core.neurons = []
    core.synapses = []
    for i, feat in enumerate(features):
        n = marble_core.Neuron(i, rep_size=features.shape[1])
        n.representation = feat.detach().cpu().numpy().astype(float)
        core.neurons.append(n)
    for src, tgt in edge_index.t().tolist():
        syn = marble_core.Synapse(int(src), int(tgt))
        core.synapses.append(syn)
        core.neurons[int(src)].synapses.append(syn)
    return core


def graph_to_module(core: Core, device: torch.device | None = None) -> GraphLayer:
    """Translate a core's graph into a ``GraphLayer`` module."""
    num = len(core.neurons)
    adj = np.zeros((num, num), dtype=np.float32)
    for s in core.synapses:
        adj[s.source, s.target] = float(s.weight)
    return GraphLayer(adj, device or torch.device("cpu"))


def module_to_graph(module: GraphLayer, params: dict) -> Core:
    """Create a ``Core`` from a ``GraphLayer``'s adjacency matrix."""
    core = Core(params)
    weight = module.weight.detach().cpu().numpy()
    n = weight.shape[0]
    core.neurons = [marble_core.Neuron(i, rep_size=core.rep_size) for i in range(n)]
    core.synapses = []
    for i in range(n):
        for j in range(weight.shape[1]):
            if weight[i, j] != 0.0:
                syn = marble_core.Synapse(i, j, weight=float(weight[i, j]))
                core.synapses.append(syn)
                core.neurons[i].synapses.append(syn)
    return core
