import os

import pytest
import torch

from remote_worker_pool import RemoteWorkerPool


def _double(t: torch.Tensor) -> torch.Tensor:
    return t * 2


def _crash_on_three(t: torch.Tensor) -> torch.Tensor:
    if not hasattr(_crash_on_three, "done"):
        _crash_on_three.done = False
    if int(t.item()) == 3 and not _crash_on_three.done:
        _crash_on_three.done = True
        raise RuntimeError("boom")
    return t + 1


def test_remote_pool_basic():
    pool = RemoteWorkerPool(num_workers=2)
    data = [torch.tensor(i) for i in range(6)]
    out = pool.map(_double, data)
    assert [int(t.item()) for t in out] == [i * 2 for i in range(6)]
    pool.shutdown()


def test_remote_pool_worker_recovery():
    pool = RemoteWorkerPool(num_workers=2, max_retries=2)
    data = [torch.tensor(i) for i in range(6)]
    out = pool.map(_crash_on_three, data)
    assert [int(t.item()) for t in out] == [i + 1 for i in range(6)]
    pool.shutdown()
