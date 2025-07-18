import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_lobes import LobeManager
from marble_core import Core
from marble_brain import Brain
from marble_neuronenblitz import Neuronenblitz
from marble_core import DataLoader
from tests.test_core_functions import minimal_params


def test_lobe_genesis_and_attention():
    params = minimal_params()
    core = Core(params)
    manager = LobeManager(core)
    manager.genesis([0, 1])
    assert len(manager.lobes) == 1
    core.neurons[0].attention_score = 1.0
    core.neurons[1].attention_score = 2.0
    manager.update_attention()
    assert manager.lobes[0].attention_score == 3.0


def test_lobe_self_attention():
    params = minimal_params()
    core = Core(params)
    manager = LobeManager(core)
    manager.genesis([0, 1])
    manager.genesis([2])
    core.neurons[0].attention_score = 0.5
    core.neurons[1].attention_score = 0.5
    core.neurons[2].attention_score = 2.0
    manager.self_attention(loss=1.0)
    assert core.neurons[0].attention_score > 0.5


def test_brain_lobe_integration():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    core.cluster_neurons(k=2)
    brain.lobe_manager.organize()
    assert brain.lobe_manager.lobes
