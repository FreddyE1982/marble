import time

from prompt_memory import PromptMemory


def test_fifo_eviction(tmp_path):
    memory = PromptMemory(max_size=2)
    memory.add("in1", "out1")
    memory.add("in2", "out2")
    memory.add("in3", "out3")
    assert len(memory) == 2
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
    assert [(r["input"], r["output"]) for r in loaded_records] == [
        ("x", "1"),
        ("y", "2"),
    ]
    assert "timestamp" in loaded_records[0]


def test_composite_with_handles_limits():
    mem = PromptMemory(max_size=3)
    mem.add("a", "1")
    mem.add("b", "2")
    composite = mem.composite_with("c", max_chars=50)
    assert (
        "Input: a" in composite
        and "Input: b" in composite
        and composite.endswith("Input: c")
    )
    # Create long strings to trigger truncation of oldest pair
    mem.add("x" * 40, "y" * 40)
    composite2 = mem.composite_with("end", max_chars=80)
    assert "x" * 40 not in composite2  # oldest removed to fit size
    assert composite2.endswith("end")


def test_load_latency_and_eviction():
    mem = PromptMemory(max_size=1000)
    start = time.time()
    for i in range(5000):
        mem.add(f"in{i}", f"out{i}")
    elapsed = time.time() - start
    avg_time = elapsed / 5000
    # ensure average insertion time remains below 10ms
    assert avg_time < 0.01
    pairs = mem.get_pairs()
    assert len(pairs) == 1000
    assert pairs[0] == ("in4000", "out4000")
    assert pairs[-1] == ("in4999", "out4999")
    prompt_start = time.time()
    prompt_text = mem.get_prompt()
    prompt_elapsed = time.time() - prompt_start
    # retrieving prompt should also remain fast
    assert prompt_elapsed < 0.5
    assert prompt_text.startswith("Input: in4000")
