import importlib

import episodic_memory


def test_store_and_query_episode(tmp_path):
    importlib.reload(episodic_memory)
    mem = episodic_memory.EpisodicMemory(
        transient_capacity=2, storage_path=tmp_path / "mem.json"
    )
    mem.add_episode({"state": 1}, reward=1.0, outcome="a")
    mem.add_episode({"state": 2}, reward=2.0, outcome="b")
    results = mem.query({"state": 2}, k=1)
    assert len(results) == 1
    assert results[0].outcome == "b"
