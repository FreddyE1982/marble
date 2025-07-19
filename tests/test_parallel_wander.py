import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_parallel_wander_train_example():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["plasticity_threshold"] = 0.0
    core = Core(params)
    nb = Neuronenblitz(core, parallel_wanderers=2, plasticity_threshold=0.0)
    before = len(core.synapses)
    out, err, path = nb.train_example(0.5, 0.2)
    assert isinstance(out, float)
    assert isinstance(err, float)
    assert path
    assert len(core.synapses) > before
