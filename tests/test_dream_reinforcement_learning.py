import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from dream_reinforcement_learning import DreamReinforcementLearner


def test_dream_reinforcement_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = DreamReinforcementLearner(core, nb, dream_cycles=1)
    err = learner.train_episode(0.5, 1.0)
    assert isinstance(err, float)
    assert learner.history
