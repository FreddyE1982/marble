import torch
from pipeline import Pipeline
from remote_hardware import load_remote_tier_plugin
from marble_core import TIER_REGISTRY


def add_tensors(a, b, *, device="cpu"):
    return (a + b).to(device)


def test_remote_step_cpu():
    tier = load_remote_tier_plugin("remote_hardware.mock_tier")
    TIER_REGISTRY[tier.name] = tier
    steps = [
        {
            "name": "add",
            "module": __name__,
            "func": "add_tensors",
            "params": {"a": torch.tensor(1), "b": torch.tensor(2)},
            "tier": tier.name,
        }
    ]
    pipe = Pipeline(steps)
    out = pipe.execute()[0]
    assert out.item() == 3
    assert out.device.type == "cpu"


def test_remote_step_gpu():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA not available")
    tier = load_remote_tier_plugin("remote_hardware.mock_tier")
    TIER_REGISTRY[tier.name] = tier
    a = torch.tensor(1, device="cuda")
    b = torch.tensor(2, device="cuda")
    steps = [
        {
            "name": "add",
            "module": __name__,
            "func": "add_tensors",
            "params": {"a": a, "b": b},
            "tier": tier.name,
        }
    ]
    pipe = Pipeline(steps)
    out = pipe.execute()[0]
    assert out.item() == 3
    assert out.device.type == "cuda"
