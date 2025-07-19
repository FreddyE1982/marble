import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_dynamic_wander_respects_dropout():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, dropout_probability=1.0)
    out, path = nb.dynamic_wander(0.5)
    assert isinstance(out, float)
    assert len(path) <= 1


def test_train_example_applies_learning_rate():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(
        core,
        learning_rate=0.5,
        gradient_noise_std=0.0,
        dropout_probability=1.0,
        structural_plasticity_enabled=False,
        weight_decay=0.0,
        weight_update_fn=lambda s, e, l: 1.0,
    )
    before = [s.weight for s in core.synapses]
    nb.train_example(0.2, 0.2)
    after = [s.weight for s in core.synapses]
    deltas = [a - b for a, b in zip(after, before)]
    assert any(pytest.approx(0.5) == d for d in deltas)


def test_weight_decay_in_train():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(
        core,
        weight_decay=0.1,
        learning_rate=0.0,
        structural_plasticity_enabled=False,
        consolidation_probability=0.0,
    )
    initial = [s.weight for s in core.synapses]
    nb.train([(0.1, 0.1)], epochs=1)
    after = [s.weight for s in core.synapses]
    for i, a in zip(initial, after):
        assert a == pytest.approx(i * 0.9)
