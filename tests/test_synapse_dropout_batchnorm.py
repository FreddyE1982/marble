import random
import numpy as np
import torch
from marble_core import Core, Synapse, SYNAPSE_TYPES
from tests.test_core_functions import minimal_params


def test_dropout_synapse_zeroes_output():
    params = minimal_params()
    core = Core(params)
    syn = Synapse(0, 1, synapse_type="dropout", dropout_prob=1.0)
    out = syn.transmit(1.0)
    assert out == 0.0


def test_batchnorm_synapse_normalises():
    params = minimal_params()
    core = Core(params)
    syn = Synapse(0, 1, synapse_type="batchnorm", momentum=1.0)
    out1 = syn.transmit(np.array([1.0, 2.0]))
    out2 = syn.transmit(np.array([3.0, 4.0]))
    mean = np.array([3.0, 4.0]).mean()
    var = np.array([3.0, 4.0]).var()
    expected = ((np.array([3.0, 4.0]) - mean) / np.sqrt(var + 1e-5))
    np.testing.assert_allclose(out2, expected)
