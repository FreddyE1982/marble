import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble_core import Core, resize_neuron_representations
from tests.test_core_functions import minimal_params


def test_resize_increases_and_decreases():
    core = Core(minimal_params())
    resize_neuron_representations(core, 8)
    for n in core.neurons:
        assert n.representation.shape == (8,)
    resize_neuron_representations(core, 2)
    for n in core.neurons:
        assert n.representation.shape == (2,)
