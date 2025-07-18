import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params


def test_message_passing_updates_representation():
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    changed = any(not np.allclose(n.representation, before[i]) for i, n in enumerate(core.neurons))
    assert changed
