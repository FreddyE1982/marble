import random
import numpy as np
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_reset_learning_state_clears_caches():
    random.seed(0)
    np.random.seed(0)
    core = Core(minimal_params())
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(core, split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb.dynamic_wander(1.0, apply_plasticity=False)
    assert nb.wander_cache
    nb.reset_learning_state()
    assert not nb.wander_cache


def test_reset_unfreezes_synapses():
    random.seed(0)
    np.random.seed(0)
    core = Core(minimal_params())
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    syn = core.add_synapse(0, 1, weight=1.0, frozen=True)
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.reset_learning_state()
    assert not syn.frozen
