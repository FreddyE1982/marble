import os
import torch

from highlevel_pipeline import HighLevelPipeline
from remote_hardware import load_remote_tier_plugin
from marble_core import TIER_REGISTRY


def add_tensors(a, b, *, device="cpu"):
    return (a + b).to(device)


def test_depends_on_reorders_execution():
    order = []

    def step_a():
        order.append("a")

    def step_b():
        order.append("b")

    pipe = HighLevelPipeline()
    pipe.add_step(step_b)
    pipe.steps[0]["name"] = "b"
    pipe.steps[0]["depends_on"] = ["a"]
    pipe.add_step(step_a)
    pipe.steps[1]["name"] = "a"
    pipe.execute()
    assert order == ["a", "b"]


def test_macro_flattens_results():
    macro = [
        {"callable": lambda: 1},
        {"callable": lambda: 2},
    ]
    pipe = HighLevelPipeline([{"name": "m", "macro": macro}])
    _, results = pipe.execute()
    assert results == [1, 2]


def test_isolated_step_runs_in_child_process():
    parent = os.getpid()

    pipe = HighLevelPipeline([{"callable": os.getpid, "isolated": True}])
    _, results = pipe.execute()
    assert results[0] != parent


def test_remote_tier_cpu():
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
    pipe = HighLevelPipeline(steps)
    _, results = pipe.execute()
    out = results[0]
    assert out.item() == 3
    assert out.device.type == "cpu"


def test_remote_tier_gpu():
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
    pipe = HighLevelPipeline(steps)
    _, results = pipe.execute()
    out = results[0]
    assert out.item() == 3
    assert out.device.type == "cuda"
