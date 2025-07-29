import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from pytorch_to_marble import convert_model, UnsupportedLayerError
from marble_core import Core
from tests.test_core_functions import minimal_params


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 2),
        )
        self.input_size = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=2)
        self.input_size = (1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_basic_conversion():
    model = SimpleModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert isinstance(core, Core)
    # expect number of neurons equals input+hidden+output+biases
    expected_neurons = 4 + 3 + 1 + 2 + 1
    assert len(core.neurons) == expected_neurons
    # verify a synapse weight from first layer
    w = model.seq[0].weight.detach().cpu().numpy()[0, 0]
    syn = core.synapses[0]
    assert syn.weight == float(w)


def test_unsupported_layer():
    class Unsupported(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    model = torch.nn.Sequential(Unsupported())
    params = minimal_params()
    with pytest.raises(UnsupportedLayerError) as exc:
        convert_model(model, core_params=params)
    assert "not supported" in str(exc.value)


def test_conv2d_conversion():
    model = ConvModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "conv2d" for n in core.neurons)

