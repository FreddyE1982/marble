import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from imitation_learning import ImitationLearner


def test_imitation_learning_step():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = ImitationLearner(core, nb, max_history=2)
    learner.record(0.1, 0.5)
    loss = learner.train_step(0.1, 0.5)
    assert isinstance(loss, float)
    assert len(learner.history) == 1
