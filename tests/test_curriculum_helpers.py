from curriculum_learning import curriculum_train
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_curriculum_train_function():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    dataset = [(0.1, 0.1), (0.5, 0.5)]
    losses = curriculum_train(core, nb, dataset, epochs=1)
    assert losses
