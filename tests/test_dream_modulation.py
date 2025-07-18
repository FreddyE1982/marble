import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from neuromodulatory_system import NeuromodulatorySystem
from tests.test_core_functions import minimal_params
import pytest


def test_compute_dream_decay():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(
        core,
        nb,
        DataLoader(),
        neuromodulatory_system=ns,
        dream_decay_arousal_scale=0.5,
        dream_decay_stress_scale=0.5,
    )
    ns.update_signals(arousal=0.6, stress=0.2)
    expected = brain.dream_synapse_decay
    expected *= 1.0 + 0.5 * 0.6
    expected *= 1.0 - 0.5 * 0.2
    expected = max(0.0, min(expected, 1.0))
    assert brain.compute_dream_decay() == pytest.approx(expected)


def test_dream_uses_dynamic_decay(monkeypatch):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(
        core,
        nb,
        DataLoader(),
        neuromodulatory_system=ns,
        dream_decay_arousal_scale=1.0,
        dream_decay_stress_scale=0.0,
    )
    syn = core.neurons[0].synapses[0]
    path = [syn]

    def fake_wander(_):
        return 0.0, path

    monkeypatch.setattr(nb, "dynamic_wander", fake_wander)
    ns.update_signals(arousal=1.0, stress=0.0)
    before = syn.weight
    decay = brain.compute_dream_decay()
    brain.dream(num_cycles=1)
    assert syn.weight == pytest.approx(before * decay)
