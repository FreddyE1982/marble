import asyncio
import time

import pytest
import torch

from highlevel_pipeline import HighLevelPipeline


async def _async_sleep_step(duration: float, device: torch.device | None = None):
    await asyncio.sleep(duration)
    return torch.ones(1, device=device)


def _sync_sleep_step(duration: float, device: torch.device | None = None):
    time.sleep(duration)
    return torch.ones(1, device=device)


def _benchmark(device: torch.device) -> tuple[float, float]:
    duration = 0.05
    hp_sync = HighLevelPipeline()
    for _ in range(3):
        hp_sync.add_step(_sync_sleep_step, params={"duration": duration, "device": device})
    start = time.perf_counter()
    hp_sync.execute()
    sync_time = time.perf_counter() - start

    hp_async = HighLevelPipeline()
    for _ in range(3):
        hp_async.add_step(_async_sleep_step, params={"duration": duration, "device": device})
    start = time.perf_counter()
    asyncio.run(hp_async.execute_async())
    async_time = time.perf_counter() - start
    return sync_time, async_time


def test_async_benchmark_cpu():
    device = torch.device("cpu")
    sync_time, async_time = _benchmark(device)
    print(f"CPU synchronous: {sync_time:.3f}s, asynchronous: {async_time:.3f}s")
    assert sync_time > 0 and async_time > 0


def test_async_benchmark_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    sync_time, async_time = _benchmark(device)
    print(f"GPU synchronous: {sync_time:.3f}s, asynchronous: {async_time:.3f}s")
    assert sync_time > 0 and async_time > 0
