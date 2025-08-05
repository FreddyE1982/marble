import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from pipeline import Pipeline
from memory_manager import MemoryManager


def make_tensor(size: int, *, device: str = "cpu"):
    return torch.zeros(size, device=device)


def make_tensor_estimate(size: int, *, device: str = "cpu"):
    return size * torch.tensor(0.0, device=device).element_size()


def test_pipeline_memory_estimation_cpu():
    mgr = MemoryManager()
    pipe = Pipeline([
        {"module": __name__, "func": "make_tensor", "params": {"size": 4}, "name": "alloc"}
    ])
    pipe.execute(memory_manager=mgr)
    assert mgr.total_reserved() == 4 * torch.tensor(0.0).element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_memory_estimation_gpu():
    mgr = MemoryManager()
    pipe = Pipeline([
        {"module": __name__, "func": "make_tensor", "params": {"size": 4}, "name": "alloc"}
    ])
    pipe.execute(memory_manager=mgr)
    assert mgr.total_reserved() == 4 * torch.tensor(0.0).element_size()
