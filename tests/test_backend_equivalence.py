import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensor_backend as tb
from core import init_seed
from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params


def test_mandelbrot_seed_consistency():
    params = minimal_params()
    params.update({"width": 4, "height": 4, "max_iter": 10, "init_noise_std": 0.0})
    tb.set_backend("numpy")
    seed_np = init_seed.generate_seed(params)
    try:
        tb.set_backend("jax")
    except ImportError:
        pytest.skip("JAX not installed")
    seed_jax = init_seed.generate_seed(params)
    assert np.allclose(seed_np, seed_jax)
    tb.set_backend("numpy")


def test_message_passing_backend_equivalence():
    params = minimal_params()
    params["init_noise_std"] = 0.0
    params["representation_noise_std"] = 0.0
    tb.set_backend("numpy")
    core_np = Core(params)
    base_reps = [n.representation.copy() for n in core_np.neurons]
    perform_message_passing(core_np)
    after_np = [n.representation.copy() for n in core_np.neurons]
    try:
        tb.set_backend("jax")
    except ImportError:
        pytest.skip("JAX not installed")
    xp = tb.xp()
    core_jax = Core(params)
    for n, rep in zip(core_jax.neurons, base_reps):
        n.representation = xp.asarray(rep)
    perform_message_passing(core_jax)
    after_jax = [tb.to_numpy(n.representation) for n in core_jax.neurons]
    for a, b in zip(after_np, after_jax):
        assert np.allclose(a, b)
    tb.set_backend("numpy")
