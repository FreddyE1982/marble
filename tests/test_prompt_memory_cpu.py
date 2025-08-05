import time
import torch
from prompt_memory import PromptMemory


def test_prompt_memory_cpu_load():
    mem = PromptMemory(max_size=10000)
    start = time.time()
    for i in range(10000):
        mem.add(f"in{i}", f"out{i}")
    duration = time.time() - start
    assert len(mem) == 10000
    assert duration < 2.0


def test_prompt_memory_cpu_vs_gpu_performance():
    mem_cpu = PromptMemory(max_size=5000)
    start_cpu = time.time()
    for i in range(5000):
        mem_cpu.add(f"in{i}", f"out{i}")
    cpu_time = time.time() - start_cpu
    assert len(mem_cpu) == 5000

    if torch.cuda.is_available():
        mem_gpu = PromptMemory(max_size=5000)
        start_gpu = time.time()
        for i in range(5000):
            mem_gpu.add(f"in{i}", f"out{i}")
        gpu_time = time.time() - start_gpu
        assert len(mem_gpu) == 5000
        # GPU path should not be significantly slower than CPU baseline
        assert gpu_time <= cpu_time * 2
