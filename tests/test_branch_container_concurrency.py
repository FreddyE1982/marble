import os
import sys
import asyncio
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from branch_container import BranchContainer

current = 0
peak = 0
lock = threading.Lock()


def sleep_step(*, device: str = "cpu"):
    global current, peak
    with lock:
        current += 1
        if current > peak:
            peak = current
    time.sleep(0.2)
    with lock:
        current -= 1
    return device


def test_gpu_concurrency_limit(monkeypatch):
    global current, peak
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(BranchContainer, "_free_gpu_memory", lambda self: 8 * 1024 ** 3)
    steps = [
        [{"module": __name__, "func": "sleep_step", "name": "a"}],
        [{"module": __name__, "func": "sleep_step", "name": "b"}],
    ]
    container = BranchContainer(steps)
    asyncio.run(container.run(None))
    assert peak >= 2
    current = 0
    peak = 0
    asyncio.run(container.run(None, max_gpu_concurrency=1))
    assert peak == 1
