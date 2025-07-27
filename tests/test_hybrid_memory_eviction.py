from hybrid_memory import HybridMemory
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_hybrid_memory_eviction(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    hm = HybridMemory(core, nb, vector_path=str(tmp_path / "vec.pkl"), symbolic_path=str(tmp_path / "sym.pkl"), max_entries=2)
    hm.store("a", 0.1)
    hm.store("b", 0.2)
    hm.store("c", 0.3)
    assert len(hm.vector_store.keys) == 2
    assert "a" not in hm.symbolic_memory.data
