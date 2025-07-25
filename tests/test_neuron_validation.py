import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Neuron


def test_invalid_stride():
    n = Neuron(0, neuron_type="conv1d")
    n.params["stride"] = 0
    with pytest.raises(ValueError):
        n.validate_params()


def test_invalid_dropout_prob():
    n = Neuron(1, neuron_type="dropout")
    n.params["p"] = 1.5
    with pytest.raises(ValueError):
        n.validate_params()


def test_negative_padding():
    n = Neuron(2, neuron_type="conv2d")
    n.params["padding"] = -1
    with pytest.raises(ValueError):
        n.validate_params()


def test_output_padding_gte_stride():
    n = Neuron(3, neuron_type="convtranspose1d")
    n.params["stride"] = 2
    n.params["output_padding"] = 2
    with pytest.raises(ValueError):
        n.validate_params()
