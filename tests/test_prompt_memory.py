import json
from prompt_memory import PromptMemory


def test_fifo_eviction(tmp_path):
    memory = PromptMemory(max_size=2)
    memory.add("in1", "out1")
    memory.add("in2", "out2")
    memory.add("in3", "out3")
    assert memory.get_pairs() == [("in2", "out2"), ("in3", "out3")]


def test_serialization_round_trip(tmp_path):
    path = tmp_path / "mem.json"
    memory = PromptMemory(max_size=3)
    memory.add("a", "1")
    memory.add("b", "2")
    memory.serialize(path)
    loaded = PromptMemory.load(path, max_size=3)
    assert loaded.get_pairs() == [("a", "1"), ("b", "2")]


def test_timestamps_persist(tmp_path):
    path = tmp_path / "mem.json"
    memory = PromptMemory(max_size=2)
    memory.add("x", "1")
    memory.add("y", "2")
    records = memory.get_records()
    assert records[0]["timestamp"] <= records[1]["timestamp"]
    memory.serialize(path)
    loaded = PromptMemory.load(path, max_size=2)
    loaded_records = loaded.get_records()
    assert [(r["input"], r["output"]) for r in loaded_records] == [("x", "1"), ("y", "2")]
    assert "timestamp" in loaded_records[0]
