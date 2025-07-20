import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from adversarial_learning import AdversarialLearner


def test_adversarial_learning_runs():
    params = minimal_params()
    core = Core(params)
    gen = Neuronenblitz(core)
    disc = Neuronenblitz(core)
    learner = AdversarialLearner(core, gen, disc, noise_dim=1)
    learner.train([0.5, 0.7], epochs=1)
    assert len(learner.history) == 2
