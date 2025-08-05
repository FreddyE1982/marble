import time

import pytest
import torch

from highlevel_pipeline import HighLevelPipeline

counter = {"calls": 0}


def _heavy_step(val: int, device: torch.device):
    counter["calls"] += 1
    time.sleep(0.01)
    return torch.full((100,), val, device=device)


def _run(device: torch.device, cache_dir: str) -> int:
    hp = HighLevelPipeline(cache_dir=cache_dir)
    for i in range(5):
        hp.add_step(_heavy_step, params={"val": i, "device": device})
    hp.execute()
    first_calls = counter["calls"]
    hp.execute()
    second_calls = counter["calls"]
    return second_calls - first_calls


def test_cache_stress_cpu(tmp_path):
    delta = _run(torch.device("cpu"), str(tmp_path))
    print(f"CPU additional calls after cache: {delta}")
    assert delta == 0


def test_cache_stress_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    delta = _run(torch.device("cuda"), str(tmp_path))
    print(f"GPU additional calls after cache: {delta}")
    assert delta == 0
