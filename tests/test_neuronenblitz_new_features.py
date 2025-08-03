import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import numpy as np
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def simple_core_two_synapses():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    s1 = core.add_synapse(0, 1, weight=1.0)
    s2 = core.add_synapse(0, 1, weight=1.0)
    return core, s1, s2


def test_gradient_path_scoring_prefers_higher_last_gradient():
    random.seed(0)
    np.random.seed(0)
    core, s1, s2 = simple_core_two_synapses()
    nb = Neuronenblitz(core, use_gradient_path_scoring=True,
                        gradient_path_score_scale=10.0,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb._prev_gradients[s1] = 0.1
    nb._prev_gradients[s2] = 1.0
    res = [(core.neurons[1], [(core.neurons[0], s1), (core.neurons[1], None)]),
           (core.neurons[1], [(core.neurons[0], s2), (core.neurons[1], None)])]
    neuron, path = nb._merge_results(res)
    assert path[0][1] is s2


def test_rms_gradient_path_scoring_prefers_higher_rms():
    random.seed(0)
    np.random.seed(0)
    core, s1, s2 = simple_core_two_synapses()
    nb = Neuronenblitz(core, use_gradient_path_scoring=True,
                        rms_gradient_path_scoring=True,
                        gradient_path_score_scale=10.0,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb._grad_sq[s1] = 0.1 ** 2
    nb._grad_sq[s2] = 1.0 ** 2
    res = [(core.neurons[1], [(core.neurons[0], s1), (core.neurons[1], None)]),
           (core.neurons[1], [(core.neurons[0], s2), (core.neurons[1], None)])]
    neuron, path = nb._merge_results(res)
    assert path[0][1] is s2


def test_activity_gating_reduces_update():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=1.0), Neuron(1, value=0.0)]
    syn = core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(core, activity_gate_exponent=1.0,
                        consolidation_probability=0.0, weight_decay=0.0)
    syn.visit_count = 10
    nb.learning_rate = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    high_visit_weight = syn.weight
    syn.weight = 1.0
    syn.visit_count = 0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    low_visit_weight = syn.weight
    assert high_visit_weight - 1.0 < low_visit_weight - 1.0


def test_subpath_cache_populates():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(core, subpath_cache_size=10, split_probability=0.0,
                        alternative_connection_prob=0.0, backtrack_probability=0.0,
                        backtrack_enabled=False)
    nb.dynamic_wander(1.0)
    assert nb.subpath_cache


def test_mixed_precision_training_runs():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(core, use_mixed_precision=True,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb.train_example(1.0, 1.0)

