import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from neuromodulatory_system import NeuromodulatorySystem
from tests.test_core_functions import minimal_params


def test_modulate_plasticity_changes_threshold():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, plasticity_threshold=5.0)
    nb.modulate_plasticity({'reward': 0.4, 'stress': 0.1})
    assert nb.plasticity_threshold < 5.0
    prev = nb.plasticity_threshold
    nb.modulate_plasticity({'reward': 0.0, 'stress': 0.6})
    assert nb.plasticity_threshold > prev


def test_brain_perform_neurogenesis():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(core, nb, DataLoader(), neuromodulatory_system=ns)
    initial_neurons = len(core.neurons)
    ns.update_signals(arousal=0.5)
    added_neurons, _ = brain.perform_neurogenesis(base_neurons=2, base_synapses=2)
    assert len(core.neurons) >= initial_neurons + added_neurons
