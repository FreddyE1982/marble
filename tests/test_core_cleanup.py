import random
from marble_core import Core
from tests.test_core_functions import minimal_params


def test_early_cleanup_removes_unused_neurons():
    params = minimal_params()
    params["early_cleanup_enabled"] = True
    params["memory_cleanup_interval"] = 0
    core = Core(params)
    core.neurons[0].synapses.clear()
    core.synapses = [s for s in core.synapses if s.source != 0 and s.target != 0]
    core.neurons[0].energy = 0.0
    before = len(core.neurons)
    core.cleanup_unused_neurons()
    assert len(core.neurons) == before - 1
