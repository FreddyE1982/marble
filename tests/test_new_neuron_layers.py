import random
import numpy as np
import math

from marble_core import Neuron, NEURON_TYPES


def test_neuron_types_list_contains_new_types():
    assert "linear" in NEURON_TYPES
    assert "conv1d" in NEURON_TYPES
    assert "conv2d" in NEURON_TYPES
    assert "batchnorm" in NEURON_TYPES
    assert "dropout" in NEURON_TYPES
    assert "relu" in NEURON_TYPES
    assert "leakyrelu" in NEURON_TYPES
    assert "elu" in NEURON_TYPES
    assert "sigmoid" in NEURON_TYPES
    assert "tanh" in NEURON_TYPES
    assert "gelu" in NEURON_TYPES
    assert "softmax" in NEURON_TYPES
    assert "maxpool1d" in NEURON_TYPES
    assert "avgpool1d" in NEURON_TYPES
    assert "flatten" in NEURON_TYPES
    assert "convtranspose1d" in NEURON_TYPES
    assert "convtranspose2d" in NEURON_TYPES
    assert "lstm" in NEURON_TYPES
    assert "gru" in NEURON_TYPES
    assert "layernorm" in NEURON_TYPES
    assert "conv3d" in NEURON_TYPES
    assert "convtranspose3d" in NEURON_TYPES
    assert "maxpool2d" in NEURON_TYPES
    assert "avgpool2d" in NEURON_TYPES
    assert "dropout2d" in NEURON_TYPES
    assert "prelu" in NEURON_TYPES
    assert "embedding" in NEURON_TYPES


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


def test_conv2d_neuron_operation():
    kern = np.array([[1.0, 0.0], [0.0, 1.0]])
    n = Neuron(0, neuron_type="conv2d")
    n.params["kernel"] = kern
    inp = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = n.process(inp)
    expected = np.array([[5.0]])
    assert np.allclose(out, expected)


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


def test_leakyrelu_neuron_operation():
    n = Neuron(0, neuron_type="leakyrelu")
    n.params["negative_slope"] = 0.1
    assert n.process(-2.0) == -0.2
    assert n.process(3.0) == 3.0


def test_elu_neuron_operation():
    n = Neuron(0, neuron_type="elu")
    n.params["alpha"] = 1.0
    out_neg = n.process(-1.0)
    out_pos = n.process(2.0)
    assert np.isclose(out_neg, math.exp(-1.0) - 1)
    assert out_pos == 2.0


def test_sigmoid_neuron_operation():
    n = Neuron(0, neuron_type="sigmoid")
    out = n.process(0.0)
    assert 0.49 < out < 0.51


def test_tanh_neuron_operation():
    n = Neuron(0, neuron_type="tanh")
    assert n.process(0.0) == 0.0


def test_gelu_neuron_operation():
    n = Neuron(0, neuron_type="gelu")
    x = np.array([-1.0, 0.0, 1.0])
    out = n.process(x)
    expected = 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    )
    assert np.allclose(out, expected)


def test_softmax_neuron_operation():
    arr = np.array([1.0, 2.0, 3.0])
    n = Neuron(0, neuron_type="softmax")
    out = n.process(arr)
    exps = np.exp(arr - np.max(arr))
    expected = exps / exps.sum()
    assert np.allclose(out, expected)


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


def test_convtranspose2d_neuron_operation():
    kern = np.array([[1.0, 0.0], [0.0, 1.0]])
    n = Neuron(0, neuron_type="convtranspose2d")
    n.params["kernel"] = kern
    inp = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = n.process(inp)
    expected = np.array([[1.0, 2.0, 0.0], [3.0, 5.0, 2.0], [0.0, 3.0, 4.0]])
    assert np.allclose(out, expected)


def test_convtranspose1d_neuron_operation():
    kern = np.array([1.0, 1.0])
    n = Neuron(0, neuron_type="convtranspose1d")
    n.params["kernel"] = kern
    inp = np.array([1.0, 2.0])
    out = n.process(inp)
    expected = np.array([1.0, 3.0, 2.0])
    assert np.allclose(out, expected)


def test_lstm_neuron_operation():
    n = Neuron(0, neuron_type="lstm")
    for k in n.params:
        n.params[k] = 1.0
    out = n.process(1.0)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    i = sig(1.0 * 1 + 0 * 1 + 1)
    f = sig(1.0 * 1 + 0 * 1 + 1)
    o = sig(1.0 * 1 + 0 * 1 + 1)
    g = math.tanh(1.0 * 1 + 0 * 1 + 1)
    c = f * 0 + i * g
    expected = o * math.tanh(c)
    assert np.isclose(out, expected, atol=1e-6)


def test_gru_neuron_operation():
    n = Neuron(0, neuron_type="gru")
    for k in n.params:
        n.params[k] = 1.0
    out = n.process(1.0)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    r = sig(1.0 * 1 + 0 * 1 + 1)
    z = sig(1.0 * 1 + 0 * 1 + 1)
    n_val = math.tanh(1.0 * 1 + r * 0 * 1 + 1)
    expected = (1 - z) * n_val + z * 0
    assert np.isclose(out, expected, atol=1e-6)


def test_layernorm_neuron_operation():
    arr = np.array([1.0, 2.0, 3.0])
    n = Neuron(0, neuron_type="layernorm")
    out = n.process(arr)
    ex = (arr - arr.mean()) / np.sqrt(arr.var() + 1e-5)
    assert np.allclose(out, ex)


def test_conv3d_neuron_operation():
    kern = np.ones((2, 2, 2))
    n = Neuron(0, neuron_type="conv3d")
    n.params["kernel"] = kern
    inp = np.ones((3, 3, 3))
    out = n.process(inp)
    expected = np.full((2, 2, 2), 8.0)
    assert np.allclose(out, expected)


def test_convtranspose3d_neuron_operation():
    kern = np.ones((2, 2, 2))
    n = Neuron(0, neuron_type="convtranspose3d")
    n.params["kernel"] = kern
    inp = np.ones((2, 2, 2))
    out = n.process(inp)
    expected = np.array(
        [
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        ],
        dtype=float,
    )
    assert np.allclose(out, expected)


def test_pool2d_neuron_operations():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    n_max = Neuron(0, neuron_type="maxpool2d")
    n_max.params["size"] = 2
    n_max.params["stride"] = 2
    out_max = n_max.process(arr)
    assert np.allclose(out_max, np.array([[4.0]]))

    n_avg = Neuron(1, neuron_type="avgpool2d")
    n_avg.params["size"] = 2
    n_avg.params["stride"] = 2
    out_avg = n_avg.process(arr)
    assert np.allclose(out_avg, np.array([[2.5]]))


def test_dropout2d_neuron_operation():
    random.seed(0)
    np.random.seed(0)
    arr = np.ones((2, 2))
    n = Neuron(0, neuron_type="dropout2d")
    n.params["p"] = 1.0
    out = n.process(arr)
    assert np.allclose(out, np.zeros_like(arr))


def test_prelu_neuron_operation():
    n = Neuron(0, neuron_type="prelu")
    n.params["alpha"] = 0.2
    assert n.process(-5.0) == -1.0
    assert n.process(3.0) == 3.0


def test_embedding_neuron_operation():
    n = Neuron(0, neuron_type="embedding")
    n.params["weights"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    n.params["num_embeddings"] = 2
    n.params["embedding_dim"] = 2
    out_single = n.process(1)
    assert np.allclose(out_single, np.array([3.0, 4.0]))
    out_multi = n.process([0, 1])
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(out_multi, expected)


def test_rnn_neuron_operation():
    n = Neuron(0, neuron_type="rnn")
    for k in n.params:
        n.params[k] = 1.0
    out1 = n.process(1.0)
    expected1 = math.tanh(1.0 * 1 + 0 * 1 + 1)
    assert np.isclose(out1, expected1, atol=1e-6)
    out2 = n.process(2.0)
    expected2 = math.tanh(2.0 * 1 + out1 * 1 + 1)
    assert np.isclose(out2, expected2, atol=1e-6)
