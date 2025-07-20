from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from harmonic_resonance_learning import HarmonicResonanceLearner


def test_harmonic_resonance_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = HarmonicResonanceLearner(core, nb, base_frequency=1.0, decay=0.9)
    err = learner.train_step(0.5, 1.0)
    assert isinstance(err, float)
    assert len(learner.history) == 1
    assert learner.base_frequency < 1.0
