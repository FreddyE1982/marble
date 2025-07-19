import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader, NEURON_TYPES
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
import random
import pytest
from neuromodulatory_system import NeuromodulatorySystem
from tests.test_core_functions import minimal_params


def test_modulate_plasticity_changes_threshold():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, plasticity_threshold=5.0)
    nb.modulate_plasticity({"reward": 0.4, "stress": 0.1})
    assert nb.plasticity_threshold < 5.0
    prev = nb.plasticity_threshold
    nb.modulate_plasticity({"reward": 0.0, "stress": 0.6})
    assert nb.plasticity_threshold > prev


def test_brain_perform_neurogenesis():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(core, nb, DataLoader(), neuromodulatory_system=ns)
    initial_neurons = len(core.neurons)
    ns.update_signals(arousal=0.5)
    added_neurons, _, n_type = brain.perform_neurogenesis(
        base_neurons=2, base_synapses=2
    )
    assert len(core.neurons) >= initial_neurons + added_neurons
    assert n_type in NEURON_TYPES


def test_neurogenesis_factor_update():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    brain.update_neurogenesis_factor(0.5)
    brain.update_neurogenesis_factor(0.6)
    assert brain.neurogenesis_factor > 1.0
    prev = brain.neurogenesis_factor
    brain.update_neurogenesis_factor(0.4)
    assert brain.neurogenesis_factor <= prev


def test_structural_plasticity_modulation():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, plasticity_threshold=0.1)
    syn = core.neurons[0].synapses[0]
    syn.potential = 1.0
    nb.modulate_plasticity({"reward": 0.5, "stress": 0.0})
    prev_count = len(core.neurons)
    nb.apply_structural_plasticity([(core.neurons[1], syn)])
    assert len(core.neurons) == prev_count + 1
    mod = 1.0 + 0.5 - 0.0
    new_syn = core.synapses[-2]
    assert new_syn.weight == pytest.approx(
        syn.weight * nb.struct_weight_multiplier1 * mod
    )


def test_preferred_neuron_type_selection():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    path = [core.neurons[0].synapses[0]]
    nb.update_attention(path, error=1.0)
    preferred = nb.get_preferred_neuron_type()
    assert preferred in NEURON_TYPES


def test_brain_neurogenesis_returns_type():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    nb.type_attention["excitatory"] = 1.0
    added, _, ntype = brain.perform_neurogenesis(base_neurons=1, base_synapses=0)
    assert added >= 1
    assert ntype == "excitatory"


def test_neurogenesis_uses_default_base_values():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(
        core,
        nb,
        DataLoader(),
        neurogenesis_base_neurons=2,
        neurogenesis_base_synapses=3,
    )
    added, syns, _ = brain.perform_neurogenesis()
    assert added >= 2
    assert syns >= 3


def test_combined_preferred_neuron_type():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    nb.type_attention["conv1d"] = 0.5
    nb.type_speed_attention["relu"] = 1.0
    preferred = nb.get_combined_preferred_neuron_type()
    assert preferred == "relu"


def test_maybe_autonomous_neurogenesis(monkeypatch):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), auto_neurogenesis_prob=1.0)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    triggered = brain.maybe_autonomous_neurogenesis(val_loss=0.5)
    assert triggered
