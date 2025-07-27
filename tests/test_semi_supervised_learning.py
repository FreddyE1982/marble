import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from semi_supervised_learning import SemiSupervisedLearner


def test_semi_supervised_learning_step():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = SemiSupervisedLearner(core, nb, unlabeled_weight=0.5)
    loss = learner.train_step((1.0, 1.0), 0.5)
    assert isinstance(loss, float)
    assert len(learner.history) == 1
