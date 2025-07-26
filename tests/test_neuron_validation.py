import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Neuron, InvalidNeuronParamsError


def test_invalid_stride():
    n = Neuron(0, neuron_type="conv1d")
    n.params["stride"] = 0
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_invalid_dropout_prob():
    n = Neuron(1, neuron_type="dropout")
    n.params["p"] = 1.5
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_negative_padding():
    n = Neuron(2, neuron_type="conv2d")
    n.params["padding"] = -1
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_output_padding_gte_stride():
    n = Neuron(3, neuron_type="convtranspose1d")
    n.params["stride"] = 2
    n.params["output_padding"] = 2
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_unknown_neuron_type():
    with pytest.raises(InvalidNeuronParamsError):
        Neuron(4, neuron_type="unknown")

def test_invalid_momentum():
    n = Neuron(5, neuron_type="batchnorm")
    n.params["momentum"] = 1.5
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_invalid_eps():
    n = Neuron(6, neuron_type="batchnorm")
    n.params["eps"] = 0
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_invalid_alpha():
    n = Neuron(7, neuron_type="elu")
    n.params["alpha"] = -0.1
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_invalid_size():
    n = Neuron(8, neuron_type="maxpool1d")
    n.params["size"] = 0
    with pytest.raises(InvalidNeuronParamsError):
        n.validate_params()


def test_invalid_rep_size():
    with pytest.raises(InvalidNeuronParamsError):
        Neuron(9, rep_size=0)
