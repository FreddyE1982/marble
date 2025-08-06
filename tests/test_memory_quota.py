import pytest
import torch

from memory_manager import MemoryManager
from pipeline import Pipeline


def allocate_large() -> torch.Tensor:
    # ~16MB tensor on float32
    return torch.zeros((1024, 1024, 4))


def test_memory_quota_exceeded():
    pipe = Pipeline(
        [
            {
                "name": "alloc",
                "func": "allocate_large",
                "module": "tests.test_memory_quota",
                "memory_limit_mb": 1,
            }
        ]
    )
    mgr = MemoryManager()
    with pytest.raises(MemoryError):
        pipe.execute(memory_manager=mgr)
    assert mgr.step_usage["alloc"] > 0
