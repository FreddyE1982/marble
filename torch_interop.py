import numpy as np
import torch

import marble_core
from marble_core import _B1, _B2, _W1, _W2, Core, configure_representation_size


class MarbleTorchAdapter(torch.nn.Module):
    """PyTorch module mirroring Marble's message passing MLP."""

    def __init__(self, core: Core) -> None:
        super().__init__()
        self.core = core
        self.w1 = torch.nn.Parameter(torch.tensor(_W1, dtype=torch.float32))
        self.b1 = torch.nn.Parameter(torch.tensor(_B1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(_W2, dtype=torch.float32))
        self.b2 = torch.nn.Parameter(torch.tensor(_B2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        h = torch.tanh(x @ self.w1 + self.b1)
        return torch.tanh(h @ self.w2 + self.b2)


def core_to_torch(core: Core) -> MarbleTorchAdapter:
    """Return a ``MarbleTorchAdapter`` wrapping ``core`` for inference."""
    return MarbleTorchAdapter(core)


def torch_to_core(model: torch.nn.Module, core: Core) -> None:
    """Update Marble's global MLP weights from a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Module containing ``w1``, ``b1``, ``w2`` and ``b2`` parameters.
    core : Core
        Core instance whose representation size will be adjusted to match
        ``model``.
    """
    rep_size = model.w1.shape[0]
    configure_representation_size(rep_size)
    marble_core._W1 = np.round(
        model.w1.detach().cpu().numpy().astype(np.float64), 8
    )
    marble_core._B1 = np.round(
        model.b1.detach().cpu().numpy().astype(np.float64), 8
    )
    marble_core._W2 = np.round(
        model.w2.detach().cpu().numpy().astype(np.float64), 8
    )
    marble_core._B2 = np.round(
        model.b2.detach().cpu().numpy().astype(np.float64), 8
    )
    for n in core.neurons:
        if n.representation.shape != (rep_size,):
            n.representation = np.zeros(rep_size, dtype=float)


def core_to_torch_graph(core: Core) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edge indices and node features as torch tensors."""
    edge_index = []
    for s in core.synapses:
        edge_index.append([s.source, s.target])
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    feats = torch.tensor([n.representation for n in core.neurons], dtype=torch.float32)
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
    edge_index = edge_index.t().tolist()
    for src, tgt in edge_index:
        syn = marble_core.Synapse(src, tgt)
        core.synapses.append(syn)
        core.neurons[src].synapses.append(syn)
    return core
