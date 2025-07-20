import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_increase_representation_size():
    params = minimal_params()
    core = Core(params)
    initial = core.rep_size
    core.increase_representation_size(2)
    assert core.rep_size == initial + 2
    for n in core.neurons:
        assert len(n.representation) == core.rep_size


def test_dimensional_search_expands():
    params = minimal_params()
    params["representation_size"] = 3
    core = Core(params)
    nb = Neuronenblitz(core)
    dim_params = {
        "enabled": True,
        "max_size": 5,
        "improvement_threshold": 1.0,
        "plateau_epochs": 1,
    }
    brain = Brain(core, nb, None, dimensional_search_params=dim_params)
    examples = [(0.1, 0.2), (0.2, 0.3)]
    brain.train(examples, epochs=3, validation_examples=examples)
    assert core.rep_size == 5
