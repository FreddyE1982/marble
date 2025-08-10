import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pytest
import torch

from highlevel_pipeline import HighLevelPipeline


async def async_step(device: str) -> torch.Tensor:
    """Asynchronous step that simulates non-blocking data loading."""
    await asyncio.sleep(0.1)
    return torch.ones(1, device=device)


def blocking_step(device: str) -> torch.Tensor:
    """Blocking step that simulates heavy computation."""
    time.sleep(0.1)
    return torch.full((1,), 2.0, device=device)


def _run_pipeline(device: str) -> list[torch.Tensor]:
    """Execute a small pipeline asynchronously and return step results."""
    hp = HighLevelPipeline(async_enabled=True)
    hp.add_step(async_step, params={"device": device})
    hp.add_step(blocking_step, params={"device": device})
    _, results = asyncio.run(hp.execute_async())
    return results


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))],
)
def test_async_multinode_execution(device: str) -> None:
    """Ensure asynchronous pipelines run concurrently across processes."""
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=2, mp_context=ctx)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    start_parallel = time.perf_counter()
    results_parallel = loop.run_until_complete(
        asyncio.gather(
            loop.run_in_executor(executor, _run_pipeline, device),
            loop.run_in_executor(executor, _run_pipeline, device),
        )
    )
    parallel_duration = time.perf_counter() - start_parallel

    start_sequential = time.perf_counter()
    _ = _run_pipeline(device)
    _ = _run_pipeline(device)
    sequential_duration = time.perf_counter() - start_sequential
    executor.shutdown()

    assert parallel_duration < sequential_duration

    for result in results_parallel:
        assert result[0].device.type == device
        assert result[1].device.type == device
        torch.testing.assert_close(result[0], torch.ones(1, device=device))
        torch.testing.assert_close(result[1], torch.full((1,), 2.0, device=device))
