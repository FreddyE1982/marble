import random
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params

def test_neuronenblitz_json_roundtrip():
    random.seed(0)
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    nb.learning_rate = 0.123
    json_str = nb.to_json()
    clone = Neuronenblitz.from_json(core, json_str)
    assert clone.learning_rate == nb.learning_rate


