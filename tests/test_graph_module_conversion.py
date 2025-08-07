import numpy as np
import torch
import marble_core
from marble_core import Core
from tests.test_core_functions import minimal_params
from torch_interop import graph_to_module, module_to_graph


def build_core_from_adj(adj: np.ndarray) -> Core:
    params = minimal_params()
    core = Core(params)
    core.neurons = [
        marble_core.Neuron(i, rep_size=core.rep_size) for i in range(adj.shape[0])
    ]
    core.synapses = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            w = float(adj[i, j])
            if w != 0.0:
                syn = marble_core.Synapse(i, j, weight=w)
                core.synapses.append(syn)
                core.neurons[i].synapses.append(syn)
    return core


def test_graph_module_roundtrip_multi_layer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj1 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    adj2 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    core1 = build_core_from_adj(adj1)
    core2 = build_core_from_adj(adj2)
    layer1 = graph_to_module(core1, device=device)
    layer2 = graph_to_module(core2, device=device)
    assert layer1.weight.device == device
    seq = torch.nn.Sequential(layer1, layer2)
    new_core1 = module_to_graph(layer1, minimal_params())
    new_core2 = module_to_graph(layer2, minimal_params())
    assert np.allclose(layer1.weight.detach().cpu().numpy(), adj1)
    assert np.allclose(layer2.weight.detach().cpu().numpy(), adj2)
    assert len(new_core1.synapses) == len(core1.synapses)
    assert len(new_core2.synapses) == len(core2.synapses)
