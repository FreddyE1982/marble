import random
import numpy as np

from marble_core import Neuron, NEURON_TYPES


def test_neuron_types_list_contains_new_types():
    assert "linear" in NEURON_TYPES
    assert "conv1d" in NEURON_TYPES
    assert "batchnorm" in NEURON_TYPES
    assert "dropout" in NEURON_TYPES
    assert "relu" in NEURON_TYPES
    assert "sigmoid" in NEURON_TYPES
    assert "tanh" in NEURON_TYPES
    assert "maxpool1d" in NEURON_TYPES
    assert "avgpool1d" in NEURON_TYPES
    assert "flatten" in NEURON_TYPES


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


def test_relu_neuron_operation():
    n = Neuron(0, neuron_type="relu")
    assert n.process(-1.0) == 0.0
    assert n.process(2.0) == 2.0


def test_sigmoid_neuron_operation():
    n = Neuron(0, neuron_type="sigmoid")
    out = n.process(0.0)
    assert 0.49 < out < 0.51


def test_tanh_neuron_operation():
    n = Neuron(0, neuron_type="tanh")
    assert n.process(0.0) == 0.0


def test_pooling_neuron_operations():
    n_max = Neuron(0, neuron_type="maxpool1d")
    n_max.params["size"] = 2
    n_max.params["stride"] = 1
    out1 = n_max.process(1.0)
    out2 = n_max.process(3.0)
    assert out2 == 3.0
    out3 = n_max.process(2.0)
    assert out3 == 3.0

    n_avg = Neuron(1, neuron_type="avgpool1d")
    n_avg.params["size"] = 2
    n_avg.params["stride"] = 1
    n_avg.process(2.0)
    out_avg = n_avg.process(4.0)
    assert out_avg == 3.0


def test_flatten_neuron_operation():
    arr = np.array([[1, 2], [3, 4]])
    n = Neuron(0, neuron_type="flatten")
    out = n.process(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
