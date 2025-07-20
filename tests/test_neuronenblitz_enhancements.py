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


def test_max_wander_depth_limit():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    # Add a self-loop to allow indefinite wandering without the depth limit
    core.add_synapse(1, 1, weight=1.0)
    nb = Neuronenblitz(
        core,
        max_wander_depth=3,
        wander_depth_noise=0.0,
        continue_decay_rate=1.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    _, path = nb.dynamic_wander(1.0)
    assert len(path) <= 3


def test_weight_update_gradient_clipping():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 0.1
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=10.0)
    assert np.isclose(syn.weight, 1.1)


def test_weight_update_with_noise():
    random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 10.0
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        gradient_noise_std=0.5,
        synapse_update_cap=2.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    np.random.seed(0)
    expected_noise = np.random.normal(0.0, 0.5)
    np.random.seed(0)
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    expected = 1.0 + 0.5 + expected_noise
    assert np.isclose(syn.weight, expected)


def test_potential_increase_capped():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        route_potential_increase=2.0,
        synapse_potential_cap=1.5,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        plasticity_threshold=100.0,
    )
    nb.dynamic_wander(1.0)
    assert np.isclose(syn.potential, 1.5)


def test_prune_low_potential_synapses():
    random.seed(0)
    np.random.seed(0)
    core, syn_main = create_simple_core()
    weak = core.add_synapse(0, 0, weight=0.01)
    weak.potential = 0.01
    nb = Neuronenblitz(
        core,
        synapse_prune_interval=1,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        plasticity_threshold=100.0,
    )
    nb.dynamic_wander(1.0)
    assert weak not in core.synapses
    assert syn_main in core.synapses


def test_update_and_get_context():
    core, _ = create_simple_core()
    nb = Neuronenblitz(core)
    nb.update_context(arousal=0.2, stress=0.1)
    ctx = nb.get_context()
    assert ctx["arousal"] == 0.2
    assert ctx["stress"] == 0.1


def test_weight_update_momentum():
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        momentum_coefficient=0.5,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert np.isclose(syn.weight, 2.25)


def test_dropout_prevents_synapse_use():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        dropout_probability=1.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    _, path = nb.dynamic_wander(1.0)
    assert len(path) == 0


def test_remote_timeout_passed():
    class DummyClient:
        def __init__(self):
            self.last_timeout = None

        def process(self, value, timeout=None):
            self.last_timeout = timeout
            return value

    core, _ = create_simple_core()
    core.neurons[1].tier = "remote"
    client = DummyClient()
    nb = Neuronenblitz(
        core,
        remote_client=client,
        remote_timeout=2.5,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.dynamic_wander(1.0)
    assert client.last_timeout == 2.5


def test_dropout_probability_decays():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        dropout_probability=0.5,
        dropout_decay_rate=0.8,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.train([(1.0, 0.0)], epochs=2)
    expected = 0.5 * 0.8 * 0.8
    assert np.isclose(nb.dropout_probability, expected)


def test_synapse_update_cap_limits_change():
    random.seed(0)
    core, syn = create_simple_core()

    def fixed_update(source, error, path_len):
        return 10.0

    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        synapse_update_cap=0.05,
        weight_update_fn=fixed_update,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert np.isclose(syn.weight, 1.05)

