import numpy as np

from marble_core import Core, Synapse
from tests.test_core_functions import minimal_params


def test_dropout_synapse_zeroes_output():
    params = minimal_params()
    Core(params)
    syn = Synapse(0, 1, synapse_type="dropout", dropout_prob=1.0)
    out = syn.transmit(1.0)
    assert out == 0.0


def test_batchnorm_synapse_normalises():
    params = minimal_params()
    Core(params)
    syn = Synapse(0, 1, synapse_type="batchnorm", momentum=1.0)
    syn.transmit(np.array([1.0, 2.0]))
    out2 = syn.transmit(np.array([3.0, 4.0]))
    mean = np.array([3.0, 4.0]).mean()
    var = np.array([3.0, 4.0]).var()
    expected = (np.array([3.0, 4.0]) - mean) / np.sqrt(var + 1e-5)
    np.testing.assert_allclose(out2, expected)


def test_add_synapse_uses_config_dropout():
    params = minimal_params()
    params["synapse_dropout_prob"] = 0.25
    core = Core(params)
    syn = core.add_synapse(0, 1, synapse_type="dropout")
    assert syn.dropout_prob == 0.25


def test_add_synapse_uses_config_batchnorm_momentum():
    params = minimal_params()
    params["synapse_batchnorm_momentum"] = 0.3
    core = Core(params)
    syn = core.add_synapse(0, 1, synapse_type="batchnorm")
    assert syn.momentum == 0.3
