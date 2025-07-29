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


class BNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3, momentum=0.2),
        )
        self.input_size = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DropoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Dropout(p=0.6),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ActivationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid(),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TanhModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Tanh(),
        )
        self.input_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class FlattenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=2),
            torch.nn.Flatten(),
        )
        self.input_size = (1, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


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
    model = torch.nn.Sequential(torch.nn.MaxPool2d(2))
    params = minimal_params()
    with pytest.raises(UnsupportedLayerError) as exc:
        convert_model(model, core_params=params)
    assert str(exc.value) == "MaxPool2d is not supported for conversion"


def test_conv2d_conversion():
    model = ConvModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "conv2d" for n in core.neurons)


def test_batchnorm_conversion():
    model = BNModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "batchnorm" for n in core.neurons)
    bn_neuron = next(n for n in core.neurons if n.neuron_type == "batchnorm")
    assert bn_neuron.params["momentum"] == 0.2


def test_dropout_conversion():
    model = DropoutModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "dropout" for n in core.neurons)
    d_neuron = next(n for n in core.neurons if n.neuron_type == "dropout")
    assert d_neuron.params["p"] == 0.6


def test_activation_conversion():
    model = ActivationModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "sigmoid" for n in core.neurons)


def test_tanh_conversion():
    model = TanhModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "tanh" for n in core.neurons)


def test_flatten_conversion():
    model = FlattenModel()
    params = minimal_params()
    core = convert_model(model, core_params=params)
    assert any(n.neuron_type == "flatten" for n in core.neurons)

