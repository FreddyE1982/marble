import random
import numpy as np

from marble_core import Neuron, NEURON_TYPES


def test_neuron_types_list_contains_new_types():
    assert "linear" in NEURON_TYPES
    assert "conv1d" in NEURON_TYPES
    assert "batchnorm" in NEURON_TYPES
    assert "dropout" in NEURON_TYPES


def test_linear_neuron_operation():
    n = Neuron(0, neuron_type="linear")
    n.params["weight"] = 2.0
    n.params["bias"] = 1.0
    out = n.process(3.0)
    assert out == 7.0


def test_conv1d_neuron_operation():
    n = Neuron(0, neuron_type="conv1d")
    n.params["kernel"] = np.array([1.0, 1.0, 1.0])
    for val in [1.0, 2.0, 3.0]:
        res = n.process(val)
    assert res == 6.0


def test_batchnorm_neuron_operation():
    n = Neuron(0, neuron_type="batchnorm")
    n.params["momentum"] = 1.0
    out1 = n.process(1.0)
    out2 = n.process(2.0)
    assert np.isfinite(out1)
    assert np.isfinite(out2)


def test_dropout_neuron_operation():
    random.seed(0)
    n = Neuron(0, neuron_type="dropout")
    n.params["p"] = 1.0
    assert n.process(5.0) == 0.0
