import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from continuous_weight_field_learning import ContinuousWeightFieldLearner


def test_cwfl_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = ContinuousWeightFieldLearner(core, nb, num_basis=3)
    loss = learner.train_step(0.1, 0.2)
    assert isinstance(loss, float)
    assert loss >= 0.0
