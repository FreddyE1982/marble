import os
import random
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensor_backend as tb

from marble_base import MetricsVisualizer
from marble_core import Core, perform_message_passing
import marble_core
from tests.test_core_functions import minimal_params


def test_message_passing_updates_representation():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    changed = any(
        not np.allclose(n.representation, before[i]) for i, n in enumerate(core.neurons)
    )
    assert changed


def test_message_passing_alpha_configurable():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    params["message_passing_alpha"] = 1.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    unchanged = all(np.allclose(b, a) for b, a in zip(before, after))
    assert unchanged


def test_message_passing_no_nan(recwarn):
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.randn(4) * 100
    perform_message_passing(core)
    assert not recwarn.list
    for n in core.neurons:
        assert np.all(np.isfinite(n.representation))


def test_energy_threshold_blocks_neurons():
    params = minimal_params()
    tb.set_backend("numpy")
    params["energy_threshold"] = 1.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
        n.energy = 0.0
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    assert all(np.allclose(b, a) for b, a in zip(before, after))


def test_representation_noise_applied():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    params["representation_noise_std"] = 0.5
    params["message_passing_alpha"] = 1.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.zeros(4)
    perform_message_passing(core)
    changed = any(not np.allclose(n.representation, 0.0) for n in core.neurons)
    assert changed


def test_message_passing_dropout():
    tb.set_backend("numpy")
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["message_passing_dropout"] = 1.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    assert all(np.allclose(b, a) for b, a in zip(before, after))


def test_attention_dropout_blocks_all_messages():
    tb.set_backend("numpy")
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["attention_dropout"] = 1.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    assert all(np.allclose(b, a) for b, a in zip(before, after))


def test_representation_activation_relu():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    params["message_passing_alpha"] = 0.0
    params["representation_activation"] = "relu"
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.uniform(-1.0, 1.0, 4)
    perform_message_passing(core)
    for n in core.neurons:
        assert np.all(n.representation >= -1e-6)


def test_message_passing_beta_effect():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    params["message_passing_beta"] = 0.0
    params["message_passing_alpha"] = 0.0
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    assert all(np.allclose(b, a) for b, a in zip(before, after))


def test_run_message_passing_iterations():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    params["message_passing_iterations"] = 2
    core_single = Core(params)
    core_multi = Core(params)
    for n1, n2 in zip(core_single.neurons, core_multi.neurons):
        rep = np.random.rand(4)
        n1.representation = rep.copy()
        n2.representation = rep.copy()

    base = [n.representation.copy() for n in core_single.neurons]
    single_change = perform_message_passing(core_single)
    multi_change = core_multi.run_message_passing()
    single_diff = sum(
        float(np.linalg.norm(n.representation - base[i]))
        for i, n in enumerate(core_single.neurons)
    )
    multi_diff = sum(
        float(np.linalg.norm(n.representation - base[i]))
        for i, n in enumerate(core_multi.neurons)
    )
    assert multi_diff > single_diff
    assert not np.isclose(multi_change, single_change)


def test_layer_norm_functionality():
    arr = np.array([[1.0, 2.0, 3.0]])
    normed = marble_core._layer_norm(arr)
    assert np.allclose(normed.mean(), 0.0)
    assert np.allclose(normed.var(), 1.0)


def test_representation_variance_metric_updated():
    tb.set_backend("numpy")
    np.random.seed(0)
    params = minimal_params()
    core = Core(params)
    mv = MetricsVisualizer()
    for n in core.neurons:
        n.representation = np.random.rand(4)
    perform_message_passing(core, metrics_visualizer=mv)
    assert mv.metrics["representation_variance"], "Metric not updated"


def test_gating_blocks_zero_signal():
    params = minimal_params()
    tb.set_backend("numpy")
    core = Core(params)
    for n in core.neurons:
        n.representation = np.random.rand(4)
    for s in core.synapses:
        s.weight = 0.0
    before = [n.representation.copy() for n in core.neurons]
    perform_message_passing(core)
    after = [n.representation.copy() for n in core.neurons]
    assert all(np.allclose(b, a) for b, a in zip(before, after))


def test_global_phase_rate_updates_phase():
    params = minimal_params()
    tb.set_backend("numpy")
    params["global_phase_rate"] = 0.5
    core = Core(params)
    initial = core.global_phase
    core.run_message_passing(iterations=2)
    assert np.isclose(core.global_phase, initial + 1.0)


def test_phase_modulation_changes_output():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["global_phase_rate"] = 0.0
    core_a = Core(params)
    core_b = Core(params)
    for n1, n2 in zip(core_a.neurons, core_b.neurons):
        rep = np.random.rand(4)
        n1.representation = rep.copy()
        n2.representation = rep.copy()
    for s1, s2 in zip(core_a.synapses, core_b.synapses):
        s1.phase = 0.0
        s2.phase = 0.0
    core_a.global_phase = 0.0
    core_b.global_phase = math.pi
    core_a.run_message_passing()
    core_b.run_message_passing()
    reps_a = [n.representation.copy() for n in core_a.neurons]
    reps_b = [n.representation.copy() for n in core_b.neurons]
    assert any(not np.allclose(a, b) for a, b in zip(reps_a, reps_b))
