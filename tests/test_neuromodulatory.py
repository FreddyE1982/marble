import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuromodulatory_system import NeuromodulatorySystem
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_neuromodulatory_update_and_context():
    ns = NeuromodulatorySystem()
    ns.update_signals(arousal=0.7, reward=0.3)
    ctx = ns.get_context()
    assert ctx['arousal'] == 0.7
    assert ctx['reward'] == 0.3


def test_brain_uses_neuromodulatory_system():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    brain = Brain(core, nb, DataLoader(), neuromodulatory_system=ns, save_dir="saved_models")
    assert brain.neuromodulatory_system is ns
    ns.update_signals(stress=0.5)
    assert brain.neuromodulatory_system.get_context()['stress'] == 0.5
