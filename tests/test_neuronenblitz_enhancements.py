import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import random
import time

import numpy as np
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


def create_chained_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0), Neuron(2, value=0.0)]
    core.synapses = []
    syn1 = core.add_synapse(0, 2, weight=1.0)
    syn2 = core.add_synapse(2, 1, weight=1.0)
    core.neurons[0].attention_score = 5.0
    core.neurons[2].attention_score = 0.0
    return core, syn1, syn2


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
    _ = np.random.normal(0.0, 0.5)
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


def test_freeze_low_impact_synapses():
    random.seed(0)
    np.random.seed(0)
    core, syn_a = create_simple_core()
    syn_b = core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb._prev_gradients[syn_a] = 1.0
    nb._prev_gradients[syn_b] = 0.0
    syn_a.visit_count = 10
    syn_b.visit_count = 0
    nb.freeze_low_impact_synapses()
    assert syn_b.frozen
    assert not syn_a.frozen


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
    # Updated expectation matches new momentum implementation with
    # eligibility traces and RMSProp scaling.
    assert np.isclose(syn.weight, 2.74292742595, atol=1e-6)


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
    _ = nb._momentum[syn]
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


def test_gradient_accumulation_steps():
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        momentum_coefficient=0.0,
        structural_plasticity_enabled=False,
        synaptic_fatigue_enabled=False,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        gradient_accumulation_steps=2,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    prev = syn.weight
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert syn.weight == pytest.approx(prev)
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    assert syn.weight != pytest.approx(prev)


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
    # Cap is applied with depth factor adjustments
    assert np.isclose(syn.weight, 1.0495049505)


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
    nb = Neuronenblitz(core, wander_cache_ttl=0.1, subpath_cache_ttl=0.0)
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
    _ = syn.weight
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


def test_shortcut_synapse_created_after_repetition():
    random.seed(0)
    np.random.seed(0)
    core, s1, s2 = create_chained_core()
    nb = Neuronenblitz(
        core,
        shortcut_creation_threshold=3,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
        continue_decay_rate=1.0,
        max_wander_depth=3,
    )
    for _ in range(3):
        nb.dynamic_wander(1.0)
    direct = [syn for syn in core.neurons[0].synapses if syn.target == 1]
    assert len(direct) == 1
    assert direct[0] not in (s1, s2)


def test_chaotic_memory_replay_generates_output():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(core)
    out, path = nb.chaotic_memory_replay(1.0, iterations=3)
    assert isinstance(out, float)
    assert path
    assert nb.chaos_state != 0.5


def test_chaotic_gating_modulates_updates():
    random.seed(0)
    np.random.seed(0)
    core, syn = create_simple_core()
    core.gradient_clip_value = 10.0
    nb = Neuronenblitz(
        core,
        consolidation_probability=0.0,
        weight_decay=0.0,
        chaotic_gating_enabled=True,
        chaotic_gating_param=3.7,
        chaotic_gate_init=0.5,
        synapse_update_cap=10.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    gate = 3.7 * 0.5 * (1 - 0.5)
    prev_v = 0.99 * 1.0 + 0.01 * (0.5**2)
    scaled = 0.5 / math.sqrt(prev_v + 1e-8)
    expected = 1.0 + scaled * gate
    assert np.isclose(syn.weight, expected, atol=1e-6)


def test_context_history_and_embedding_bias_choice():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0), Neuron(2, value=0.0)]
    core.neurons[1].representation = np.array([0.0, 0.0, 1.0, 0.0])
    core.neurons[2].representation = np.array([1.0, 0.0, 0.0, 0.0])
    syn1 = core.add_synapse(0, 1, weight=1.0)
    syn2 = core.add_synapse(0, 2, weight=1.0)
    nb = Neuronenblitz(
        core,
        dynamic_attention_enabled=False,
        synaptic_fatigue_enabled=False,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        context_history_size=5,
        context_embedding_decay=1.0,
    )
    nb.update_context(reward=1.0)
    choice = nb.weighted_choice([syn1, syn2])
    assert choice is syn1


def test_context_history_size_limit_and_embedding():
    core, _ = create_simple_core()
    nb = Neuronenblitz(core, context_history_size=2, context_embedding_decay=1.0)
    nb.update_context(reward=1.0)
    nb.update_context(stress=1.0)
    nb.update_context(arousal=1.0)
    assert len(nb.context_history) == 2
    emb = nb.get_context_embedding()
    assert np.allclose(emb, [0.5, 1.0, 1.0])


def test_emergent_connection_creation(monkeypatch):
    random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        emergent_connection_prob=1.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    monkeypatch.setattr(nb, "decide_synapse_action", lambda: None)
    prev = len(core.synapses)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    nb.dynamic_wander(1.0)
    assert len(core.synapses) == prev + 1


def test_cosine_scheduler_steps_learning_rate():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        lr_scheduler="cosine",
        scheduler_steps=2,
        min_learning_rate=0.1,
        max_learning_rate=1.0,
    )
    nb.learning_rate = nb.max_learning_rate
    lr_start = nb.learning_rate
    nb.step_lr_scheduler()
    mid_lr = nb.learning_rate
    nb.step_lr_scheduler()
    end_lr = nb.learning_rate
    assert lr_start > end_lr
    assert mid_lr >= end_lr


def test_exponential_scheduler_decreases_learning_rate():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        lr_scheduler="exponential",
        scheduler_gamma=0.5,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
    )
    nb.learning_rate = nb.max_learning_rate
    nb.step_lr_scheduler()
    assert nb.learning_rate == 0.5
    nb.step_lr_scheduler()
    assert nb.learning_rate == 0.25


def test_cyclic_scheduler_cycles_learning_rate():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        lr_scheduler="cyclic",
        scheduler_steps=2,
        min_learning_rate=0.1,
        max_learning_rate=1.0,
    )
    nb.learning_rate = nb.min_learning_rate
    nb.step_lr_scheduler()
    start_lr = nb.learning_rate
    nb.step_lr_scheduler()
    mid_lr = nb.learning_rate
    nb.step_lr_scheduler()
    end_lr = nb.learning_rate
    assert start_lr == pytest.approx(0.1)
    assert mid_lr == pytest.approx(1.0)
    assert end_lr == pytest.approx(start_lr)


def test_cosine_epsilon_scheduler():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        epsilon_scheduler="cosine",
        epsilon_scheduler_steps=2,
        rl_epsilon=1.0,
        rl_min_epsilon=0.1,
    )
    start = nb.rl_epsilon
    nb.step_epsilon_scheduler()
    mid = nb.rl_epsilon
    nb.step_epsilon_scheduler()
    end = nb.rl_epsilon
    assert start > end
    assert mid >= end


def test_exponential_epsilon_scheduler():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        epsilon_scheduler="exponential",
        epsilon_scheduler_gamma=0.5,
        rl_epsilon=1.0,
        rl_min_epsilon=0.1,
    )
    nb.step_epsilon_scheduler()
    assert nb.rl_epsilon == 0.5
    nb.step_epsilon_scheduler()
    assert nb.rl_epsilon == 0.25


def test_cyclic_epsilon_scheduler():
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        epsilon_scheduler="cyclic",
        epsilon_scheduler_steps=2,
        rl_epsilon=1.0,
        rl_min_epsilon=0.1,
    )
    nb.step_epsilon_scheduler()
    start_eps = nb.rl_epsilon
    nb.step_epsilon_scheduler()
    mid_eps = nb.rl_epsilon
    nb.step_epsilon_scheduler()
    end_eps = nb.rl_epsilon
    assert start_eps == pytest.approx(1.0)
    assert mid_eps == pytest.approx(nb.rl_min_epsilon)
    assert end_eps == pytest.approx(start_eps)


def test_experience_replay_buffer_fills_and_replays():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        use_experience_replay=True,
        replay_buffer_size=5,
        replay_batch_size=2,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.train([(1.0, 0.0)] * 3, epochs=1)
    assert len(nb.replay_buffer) >= 3
    prev_len = len(nb.replay_buffer)
    nb.train([(1.0, 0.0)] * 2, epochs=1)
    assert len(nb.replay_buffer) >= prev_len


def test_replay_importance_weights():
    random.seed(0)
    np.random.seed(0)
    core, _ = create_simple_core()
    nb = Neuronenblitz(
        core,
        use_experience_replay=True,
        replay_buffer_size=10,
        replay_alpha=1.0,
        replay_beta=1.0,
    )
    nb.replay_buffer.extend([(0.0, 0.0, [], [], {}) for _ in range(2)])
    nb.replay_priorities.extend([1.0, 3.0])
    samples = nb.sample_replay_batch(2)
    samples.sort(key=lambda x: x[0])
    weights = [w for _, w in samples]
    pri = np.array([1.0, 3.0])
    probs = pri / pri.sum()
    expected = (1 / (len(pri) * probs)) ** 1.0
    expected = expected / expected.max()
    assert np.allclose(weights, expected)


def test_train_example_uses_sample_weight():
    random.seed(0)
    np.random.seed(0)
    core1, syn1 = create_simple_core()
    nb1 = Neuronenblitz(
        core1,
        consolidation_probability=0.0,
        weight_decay=0.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        synaptic_fatigue_enabled=False,
        plasticity_threshold=100.0,
    )
    nb1.learning_rate = 1.0
    core1.neurons[0].value = 1.0
    nb1.train_example(1.0, 0.0, sample_weight=1.0)
    base_update = syn1.weight - 1.0

    random.seed(0)
    np.random.seed(0)
    core2, syn2 = create_simple_core()
    nb2 = Neuronenblitz(
        core2,
        consolidation_probability=0.0,
        weight_decay=0.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        synaptic_fatigue_enabled=False,
        plasticity_threshold=100.0,
    )
    nb2.learning_rate = 1.0
    core2.neurons[0].value = 1.0
    nb2.train_example(1.0, 0.0, sample_weight=2.0)
    weighted_update = syn2.weight - 1.0
    assert np.isclose(weighted_update, 2 * base_update, rtol=1e-2)


def test_memory_gate_biases_selection():
    random.seed(0)
    np.random.seed(0)
    core, syn_a = create_simple_core()
    _ = core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(
        core,
        memory_gate_strength=1.0,
        episodic_memory_threshold=0.2,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    nb.train_example(1.0, 1.0)  # should record successful path
    assert nb.memory_gates


def test_memory_gates_modulate_attention():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0), Neuron(2, value=0.0)]
    core.synapses = []
    syn_a = core.add_synapse(0, 1, weight=1.0)
    syn_b = core.add_synapse(0, 2, weight=1.0)
    nb = Neuronenblitz(
        core,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        consolidation_probability=0.0,
        weight_decay=0.0,
        gradient_accumulation_steps=100,
    )
    nb.memory_gates[syn_a] = 1.0
    nb.apply_weight_updates_and_attention([syn_a], error=1.0)
    nb.apply_weight_updates_and_attention([syn_b], error=1.0)
    assert core.neurons[1].attention_score > core.neurons[2].attention_score


def test_curiosity_strength_biases_toward_novel_synapse():
    random.seed(0)
    np.random.seed(0)
    core, syn_old = create_simple_core()
    syn_new = core.add_synapse(0, 1, weight=1.0)
    syn_old.visit_count = 10
    nb = Neuronenblitz(
        core,
        curiosity_strength=5.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
        backtrack_probability=0.0,
        backtrack_enabled=False,
    )
    choices = [nb.weighted_choice(core.neurons[0].synapses) for _ in range(50)]
    assert choices.count(syn_new) > choices.count(syn_old)


def test_depth_clip_scaling_reduces_update_for_deep_paths():
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        depth_clip_scaling=1.0,
        synapse_update_cap=1.0,
        consolidation_probability=0.0,
        weight_decay=0.0,
    )
    nb.learning_rate = 1.0
    core.neurons[0].value = 1.0
    other = core.add_synapse(0, 1, weight=1.0)
    path_deep = [other, other, other, other, syn]
    nb.apply_weight_updates_and_attention(path_deep, error=1.0)
    weight_deep = syn.weight
    syn.weight = 1.0
    nb.apply_weight_updates_and_attention([syn], error=1.0)
    weight_shallow = syn.weight
    assert weight_deep - 1.0 < weight_shallow - 1.0


def test_active_forgetting_decays_context():
    core, _ = create_simple_core()
    nb = Neuronenblitz(core, forgetting_rate=0.5)
    nb.update_context(reward=1.0)
    nb.dynamic_wander(1.0)
    assert nb.context_history[0]["reward"] < 1.0


def test_structural_dropout_skips_plasticity(monkeypatch):
    random.seed(0)
    core, syn = create_simple_core()
    nb = Neuronenblitz(
        core,
        structural_dropout_prob=1.0,
        structural_plasticity_enabled=True,
        plasticity_threshold=0.0,
        split_probability=0.0,
        alternative_connection_prob=0.0,
    )
    syn.potential = nb.plasticity_threshold + 1.0
    prev = len(core.synapses)
    nb.apply_structural_plasticity([(core.neurons[0], syn)])
    assert len(core.synapses) == prev
