from marble_interface import curriculum_train_marble_system
from marble_main import MARBLE
from tests.test_core_functions import minimal_params


def test_interface_curriculum_train():
    params = minimal_params()
    m = MARBLE(params)
    dataset = [(0.2, 0.2), (1.0, 1.0)]
    losses = curriculum_train_marble_system(m, dataset, epochs=1)
    assert isinstance(losses, list) and len(losses) > 0
