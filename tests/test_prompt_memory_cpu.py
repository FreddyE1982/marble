import time
from prompt_memory import PromptMemory


def test_prompt_memory_cpu_load():
    mem = PromptMemory(max_size=10000)
    start = time.time()
    for i in range(10000):
        mem.add(f"in{i}", f"out{i}")
    duration = time.time() - start
    assert len(mem) == 10000
    assert duration < 2.0
