import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def create_simple_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    syn = core.add_synapse(0, 1, weight=1.0)
    return core, syn


def test_synaptic_fatigue_accumulates():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        synaptic_fatigue_enabled=True,
        fatigue_increase=0.5,
        fatigue_decay=0.9,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        plasticity_threshold=100.0,
    )
    nb.dynamic_wander(1.0)
    first = syn.fatigue
    nb.dynamic_wander(1.0)
    second = syn.fatigue
    assert first > 0
    assert second > first


def test_learning_rate_adjustment_bounds():
    random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        lr_adjustment_factor=0.5,
        min_learning_rate=0.01,
        max_learning_rate=1.0,
    )
    nb.learning_rate = 0.1
    nb.error_history.extend([0.2] * 5 + [0.3] * 5)
    nb.adjust_learning_rate()
    assert nb.learning_rate > 0.1
    prev = nb.learning_rate
    nb.error_history.extend([0.1] * 5)
    nb.adjust_learning_rate()
    assert nb.learning_rate < prev
    assert nb.min_learning_rate <= nb.learning_rate <= nb.max_learning_rate

