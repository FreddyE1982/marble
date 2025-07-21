import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from neural_schema_induction import NeuralSchemaInductionLearner


def test_neural_schema_induction_creates_schema():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = NeuralSchemaInductionLearner(core, nb, support_threshold=2, max_schema_size=2)
    initial = len(core.neurons)
    for _ in range(5):
        learner.train_step(0.5)
    assert len(core.neurons) > initial
    assert learner.schemas
