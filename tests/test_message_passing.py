import os, sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params


def test_message_passing_updates_representation():
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


def test_representation_activation_relu():
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
