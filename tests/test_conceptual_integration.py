import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from conceptual_integration import ConceptualIntegrationLearner


def test_conceptual_integration_adds_neuron():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = ConceptualIntegrationLearner(core, nb, blend_probability=1.0, similarity_threshold=0.1)
    initial = len(core.neurons)
    learner.train_step(0.5)
    learner.train_step(0.6)
    assert len(core.neurons) > initial

