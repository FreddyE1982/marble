import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from marble_core import Core, Synapse
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_nan_detection():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    # inject NaN weight
    core.synapses.append(Synapse(0, 0, weight=np.nan))
    with pytest.raises(ValueError):
        nb.dynamic_wander(0.1)


def test_core_finite_check():
    params = minimal_params()
    core = Core(params)
    core.neurons[0].representation[0] = float("inf")
    with pytest.raises(ValueError):
        core.check_finite_state()
