import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import MemorySystem, Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from neuromodulatory_system import NeuromodulatorySystem
from tests.test_core_functions import minimal_params


def test_memory_system_roundtrip(tmp_path):
    ms = MemorySystem(long_term_path=tmp_path / "lt.pkl")
    layer = ms.choose_layer({})
    layer.store("a", 1)
    assert ms.short_term.retrieve("a") == 1
    ms.consolidate()
    assert ms.short_term.retrieve("a") is None
    assert ms.long_term.retrieve("a") == 1


def test_brain_memory_integration(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    ns = NeuromodulatorySystem()
    ms = MemorySystem(long_term_path=tmp_path / "brain_lt.pkl")
    brain = Brain(core, nb, DataLoader(), neuromodulatory_system=ns, memory_system=ms)

    brain.store_memory("key", 42)
    assert brain.retrieve_memory("key") == 42
    brain.consolidate_memory()
    brain.store_memory("other", 7)
    assert brain.retrieve_memory("other") == 7
    brain.consolidate_memory()
    assert brain.retrieve_memory("key") == 42
    assert brain.retrieve_memory("other") == 7
