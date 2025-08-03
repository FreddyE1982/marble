from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.memory import bias_with_episodic_memory, decay_memory_gates
from tests.test_core_functions import minimal_params


def test_decay_memory_gates():
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    syn = core.add_synapse(0, 1, weight=0.1)
    nb.memory_gates[syn] = 1.0
    decay_memory_gates(nb)
    assert syn not in nb.memory_gates or nb.memory_gates[syn] < 1.0


def test_bias_with_episodic_memory_follows_path():
    core = Core(minimal_params())
    nb = Neuronenblitz(
        core,
        episodic_memory_prob=1.0,
        episodic_sim_length=1,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    syn = core.add_synapse(0, 1, weight=1.0)
    nb.episodic_memory.append([syn])
    entry = core.neurons[0]
    entry.value = 1.0
    current, path, remaining = bias_with_episodic_memory(
        nb, entry, [(entry, None)], 1
    )
    assert path[-1][1] == syn
    assert remaining == 0
