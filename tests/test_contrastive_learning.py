import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from contrastive_learning import ContrastiveLearner


def test_contrastive_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = ContrastiveLearner(core, nb, temperature=0.5)
    loss = learner.train([0.1, 0.2])
    assert isinstance(loss, float)
    assert loss >= 0.0
