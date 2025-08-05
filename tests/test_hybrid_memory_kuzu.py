import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from hybrid_memory import HybridMemory
from topology_kuzu import TopologyKuzuTracker
from tests.test_core_functions import minimal_params


def test_kuzu_memory_separate_from_topology(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    topo_path = tmp_path / "topology.kuzu"
    mem_path = tmp_path / "memory.kuzu"
    TopologyKuzuTracker(core, str(topo_path))
    hm = HybridMemory(core, nb, kuzu_path=str(mem_path))
    hm.store("a", 1.0)
    res = hm.retrieve(1.0, top_k=1)
    assert res and res[0][0] == "a"
    assert topo_path.exists() and mem_path.exists()
    assert topo_path != mem_path
