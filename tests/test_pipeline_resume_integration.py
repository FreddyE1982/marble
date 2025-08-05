import torch
import pytest

from highlevel_pipeline import HighLevelPipeline


call_counter = {"step1": 0, "step2": 0}


def step1(device: str = "cpu"):
    call_counter["step1"] += 1
    return torch.tensor(1, device=device)


def step2(device: str = "cpu"):
    call_counter["step2"] += 1
    return torch.tensor(2, device=device)


def test_interrupted_resume_cpu(tmp_path):
    pipe = HighLevelPipeline(dataset_version="cpu_v1", cache_dir=str(tmp_path / "cpu_cache"))
    pipe.add_step(step1, params={"device": "cpu"})
    pipe.add_step(step2, params={"device": "cpu"})

    # simulate interruption after first step
    pipe.execute_until(0)
    path = tmp_path / "cpu_chk.pkl"
    pipe.save_checkpoint(path)

    call_counter["step1"] = 0
    call_counter["step2"] = 0

    loaded = HighLevelPipeline.load_checkpoint(str(path))
    _, results = loaded.execute()

    assert call_counter["step1"] == 0
    assert call_counter["step2"] == 1
    assert results[-1].item() == 2
    assert loaded.dataset_version == "cpu_v1"


def test_interrupted_resume_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # baseline result on CPU
    base = HighLevelPipeline(dataset_version="gpu_v1", cache_dir=str(tmp_path / "base_cache"))
    base.add_step(step1, params={"device": "cpu"})
    base.add_step(step2, params={"device": "cpu"})
    _, cpu_results = base.execute()
    expected = cpu_results[-1].cpu()

    pipe = HighLevelPipeline(dataset_version="gpu_v1", cache_dir=str(tmp_path / "gpu_cache"))
    pipe.add_step(step1, params={"device": "cuda"})
    pipe.add_step(step2, params={"device": "cuda"})
    pipe.execute_until(0)
    path = tmp_path / "gpu_chk.pkl"
    pipe.save_checkpoint(path)

    call_counter["step1"] = 0
    call_counter["step2"] = 0

    loaded = HighLevelPipeline.load_checkpoint(str(path))
    _, res = loaded.execute()

    assert call_counter["step1"] == 0
    assert call_counter["step2"] == 1
    assert torch.allclose(res[-1].cpu(), expected)
    assert loaded.dataset_version == "gpu_v1"

