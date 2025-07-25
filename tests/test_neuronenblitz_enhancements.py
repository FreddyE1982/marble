import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import time
import math
import pytest
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
    assert np.isclose(syn.weight, 1.10049870545, atol=1e-6)


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
    expected = 2.37578056622
    assert np.isclose(syn.weight, expected, atol=1e-6)


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
    assert np.isclose(syn.weight, 2.75282841605, atol=1e-6)


def test_weight_update_scales_with_fatigue():
    random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 10.0
    syn.fatigue = 0.5
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        synaptic_fatigue_enabled=True,
        momentum_coefficient=0.5,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert np.isclose(syn.weight, 1.37641420803, atol=1e-6)


def test_momentum_values_decay():
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
    first = nb._momentum[syn]
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    second = nb._momentum[syn]
    assert second == pytest.approx(1.17986383427, abs=1e-6)


def test_eligibility_traces_accumulate():
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        momentum_coefficient=0.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert np.isclose(syn.weight, 2.45470864944, atol=1e-6)


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
    assert 0.0 <= nb.dropout_probability < 0.5


def test_adjust_dropout_rate_increases_and_decreases():
    core, _ = create_simple_core()
    nb = Neuronenblitz(core, dropout_probability=0.2, dropout_decay_rate=0.9)
    nb.adjust_dropout_rate(1.0)
    assert nb.dropout_probability > 0.2
    prev = nb.dropout_probability
    nb.adjust_dropout_rate(0.0)
    assert nb.dropout_probability < prev


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


def test_beam_wander_selects_best_path():
    random.seed(0)
    core, syn = create_simple_core()
    core.add_synapse(0, 1, weight=0.5)
    nb = Neuronenblitz(
        core,
        beam_width=2,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    out, path = nb.dynamic_wander(1.0)
    assert isinstance(out, float)
    assert path


def test_beam_wander_penalizes_fatigue():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.synapses = []
    syn_fatigued = core.add_synapse(0, 1, weight=1.0)
    syn_fatigued.fatigue = 0.9
    syn_fresh = core.add_synapse(0, 1, weight=1.0)
    syn_fresh.fatigue = 0.0
    nb = Neuronenblitz(
        core,
        beam_width=2,
        max_wander_depth=1,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    _, path = nb.dynamic_wander(1.0)
    assert path
    assert path[0] is syn_fresh


def test_entry_neuron_selection_bias():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    core.neurons[0].attention_score = 5.0
    core.neurons[1].attention_score = 0.0
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    selections = [nb._select_entry_neuron().id for _ in range(50)]
    assert selections.count(0) > selections.count(1)


def test_weighted_choice_prefers_attention_and_low_fatigue():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0), Neuron(2, value=0.0)]
    core.synapses = []
    syn_a = core.add_synapse(0, 1, weight=1.0)
    syn_b = core.add_synapse(0, 2, weight=1.0)
    syn_a.fatigue = 0.0
    syn_b.fatigue = 0.8
    core.neurons[1].attention_score = 2.0
    core.neurons[2].attention_score = 0.0
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
        synaptic_fatigue_enabled=True,
    )
    selections = [nb.weighted_choice(core.neurons[0].synapses) for _ in range(50)]
    assert selections.count(syn_a) > selections.count(syn_b)


def test_visit_count_decay_and_bias():
    random.seed(0)
    np.random.seed(0)
    core, syn_a = create_simple_core()
    syn_b = core.add_synapse(0, 1, weight=1.0)
    syn_a.visit_count = 10
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.decay_visit_counts(decay=0.5)
    assert syn_a.visit_count == 5
    selections = [nb.weighted_choice(core.neurons[0].synapses) for _ in range(50)]
    assert selections.count(syn_b) > selections.count(syn_a)


def test_weighted_choice_handles_large_scores():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    syn_a = core.add_synapse(0, 1, weight=1e9)
    syn_b = core.add_synapse(0, 1, weight=1e9 + 1)
    nb = Neuronenblitz(core)
    choice = nb.weighted_choice([syn_a, syn_b])
    assert choice in (syn_a, syn_b)


def test_dynamic_wander_caches_results():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(core)
    out1, path1 = nb.dynamic_wander(1.0, apply_plasticity=False)
    syn.weight = 2.0
    out2, path2 = nb.dynamic_wander(1.0, apply_plasticity=False)
    assert out1 == out2
    assert path1 == path2


def test_dynamic_wander_cache_size_limit():
    random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(core)
    for i in range(nb._cache_max_size + 10):
        nb.dynamic_wander(float(i), apply_plasticity=False)
    assert len(nb.wander_cache) == nb._cache_max_size


def test_dynamic_wander_cache_ttl_expiration():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(core, wander_cache_ttl=0.1)
    out1, path1 = nb.dynamic_wander(1.0, apply_plasticity=False)
    time.sleep(0.2)
    syn.weight = 2.0
    out2, path2 = nb.dynamic_wander(1.0, apply_plasticity=False)
    assert out1 != out2 or path1 != path2


def test_rmsprop_adaptive_scaling():
    random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 10.0
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        synapse_update_cap=10.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=5.0)
    first_v = nb._grad_sq[syn]
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=5.0)
    second_v = nb._grad_sq[syn]
    assert second_v > first_v


def test_gradient_alignment_gating():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 10.0
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        momentum_coefficient=0.0,
        gradient_noise_std=0.0,
        synapse_update_cap=10.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    first_weight = syn.weight
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=-1.0)
    assert syn.weight == pytest.approx(1.02383970547, abs=1e-6)


def test_phase_gated_updates_modulate_direction():
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        phase_rate=0.0,
        phase_adaptation_rate=0.5,
        momentum_coefficient=0.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    syn.phase = 0.0
    nb.global_phase = 0.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    pos_weight = syn.weight
    syn.phase = math.pi
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert syn.weight < pos_weight
