import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble import DataLoader
from marble_autograd import MarbleAutogradLayer
from marble_brain import Brain
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_autograd_forward_and_backward():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    layer = MarbleAutogradLayer(brain, learning_rate=0.1)

    x = torch.tensor([0.1, 0.2], requires_grad=True)
    weights_before = [syn.weight for syn in core.synapses]
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    weights_after = [syn.weight for syn in core.synapses]
    assert any(a != b for a, b in zip(weights_after, weights_before))
