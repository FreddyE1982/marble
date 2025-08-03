import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


calls: list[tuple[int, float]] = []
orig_apply_weight_updates = Neuronenblitz.apply_weight_updates_and_attention


def record_apply_weight_updates(nb, path, error):
    calls.append((len(path), error))
    return orig_apply_weight_updates(nb, path, error)


def test_parallel_wander_train_example():
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["plasticity_threshold"] = 0.0
    core = Core(params)
    nb = Neuronenblitz(core, parallel_wanderers=2, plasticity_threshold=0.0)
    before = len(core.synapses)
    out, err, path = nb.train_example(0.5, 0.2)
    assert isinstance(out, float)
    assert isinstance(err, float)
    assert path
    assert len(core.synapses) > before


def test_parallel_average_applies_updates(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    params = minimal_params()
    params["plasticity_threshold"] = 0.0
    core = Core(params)
    nb = Neuronenblitz(
        core,
        parallel_wanderers=2,
        plasticity_threshold=0.0,
        parallel_update_strategy="average",
    )
    global calls
    calls = []
    monkeypatch.setattr(
        Neuronenblitz, "apply_weight_updates_and_attention", record_apply_weight_updates
    )
    out, err, path = nb.train_example(0.5, 0.2)
    monkeypatch.setattr(
        Neuronenblitz, "apply_weight_updates_and_attention", orig_apply_weight_updates
    )
    assert len(calls) == 2
    assert isinstance(out, float)
    assert isinstance(err, float)
    assert path
