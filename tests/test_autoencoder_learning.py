from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from autoencoder_learning import AutoencoderLearner


def test_autoencoder_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = AutoencoderLearner(core, nb, noise_std=0.1)
    loss = learner.train_step(0.5)
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert len(learner.history) == 1
