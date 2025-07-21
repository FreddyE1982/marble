import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_brain import Brain
from marble_core import Core, DataLoader, MemorySystem
from marble_neuronenblitz import Neuronenblitz
from neuromodulatory_system import NeuromodulatorySystem
from tests.test_core_functions import minimal_params


def test_memory_system_roundtrip(tmp_path):
    ms = MemorySystem(long_term_path=tmp_path / "lt.pkl")
    ms.store("a", 1, context={})
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


def test_auto_consolidation(tmp_path):
    ms = MemorySystem(long_term_path=tmp_path / "auto.pkl", consolidation_interval=2)
    ms.store("k1", 1)
    ms.store("k2", 2)
    assert ms.long_term.retrieve("k1") == 1
    assert ms.long_term.retrieve("k2") == 2


def test_memory_system_threshold():
    ms = MemorySystem(threshold=0.3)
    layer = ms.choose_layer({"arousal": 0.4})
    assert layer is ms.long_term
    layer = ms.choose_layer({"arousal": 0.2})
    assert layer is ms.short_term
