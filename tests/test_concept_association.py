import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


def test_concept_neuron_created_after_threshold():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(
        core,
        concept_association_threshold=2,
        concept_learning_rate=1.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        structural_plasticity_enabled=False,
    )
    start = len(core.neurons)
    nb.dynamic_wander(1.0)
    nb.dynamic_wander(1.0)
    assert len(core.neurons) > start
