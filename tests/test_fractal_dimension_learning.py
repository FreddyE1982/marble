import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from fractal_dimension_learning import FractalDimensionLearner


def test_fractal_dimension_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = FractalDimensionLearner(core, nb, target_dimension=1.0)
    err = learner.train_step(0.5, 1.0)
    assert isinstance(err, float)
    assert learner.history
