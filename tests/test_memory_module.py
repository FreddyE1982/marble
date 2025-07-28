from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.memory import decay_memory_gates
from tests.test_core_functions import minimal_params


def test_decay_memory_gates():
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    syn = core.add_synapse(0, 1, weight=0.1)
    nb.memory_gates[syn] = 1.0
    decay_memory_gates(nb)
    assert syn not in nb.memory_gates or nb.memory_gates[syn] < 1.0
