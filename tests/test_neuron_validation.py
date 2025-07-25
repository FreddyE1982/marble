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
