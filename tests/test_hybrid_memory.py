import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from hybrid_memory import HybridMemory
from tests.test_core_functions import minimal_params


def test_hybrid_memory_store_and_retrieve(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    hm = HybridMemory(core, nb,
                      vector_path=tmp_path / "vec.pkl",
                      symbolic_path=tmp_path / "sym.pkl")
    hm.store("a", 1.0)
    results = hm.retrieve(1.0, top_k=1)
    assert results and results[0][0] == "a"


def test_hybrid_memory_forget(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    hm = HybridMemory(core, nb,
                      vector_path=tmp_path / "vec.pkl",
                      symbolic_path=tmp_path / "sym.pkl")
    for i in range(5):
        hm.store(f"k{i}", float(i))
    hm.forget_old(max_entries=3)
    assert len(hm.vector_store.keys) == 3
