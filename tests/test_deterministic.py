import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensor_backend as tb
from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params


def test_message_passing_deterministic():
    seed = 123
    tb.set_backend("numpy")
    p1 = minimal_params()
    p1['random_seed'] = seed
    core1 = Core(p1)
    p2 = minimal_params()
    p2['random_seed'] = seed
    core2 = Core(p2)
    perform_message_passing(core1)
    perform_message_passing(core2)
    reps1 = [n.representation.copy() for n in core1.neurons]
    reps2 = [n.representation.copy() for n in core2.neurons]
    for a, b in zip(reps1, reps2):
        assert np.allclose(a, b)
