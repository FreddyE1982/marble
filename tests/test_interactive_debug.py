import pytest
import torch

from pipeline import Pipeline


def add_value(tensor: torch.Tensor, value: float, device: str) -> torch.Tensor:
    return tensor.to(device) + value


def test_interactive_debug_cpu():
    pipe = Pipeline()
    pipe.add_step(
        "add_value",
        module="tests.test_interactive_debug",
        params={"tensor": torch.ones(2), "value": 5, "device": "cpu"},
    )
    debugger = pipe.enable_interactive_debugging(interactive=False)
    result = pipe.execute()[0]
    name = pipe.steps[0]["name"]
    assert debugger.inputs[name]["params"]["tensor"]["device"] == "cpu"
    assert debugger.outputs[name]["device"] == result.device.type
    assert torch.allclose(result, torch.tensor([6.0, 6.0], device=result.device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_interactive_debug_gpu():
    pipe = Pipeline()
    pipe.add_step(
        "add_value",
        module="tests.test_interactive_debug",
        params={"tensor": torch.ones(2), "value": 5, "device": "cuda"},
    )
    debugger = pipe.enable_interactive_debugging(interactive=False)
    result = pipe.execute()[0]
    name = pipe.steps[0]["name"]
    assert debugger.inputs[name]["params"]["tensor"]["device"] == "cpu"
    assert debugger.outputs[name]["device"] == "cuda"
    assert result.device.type == "cuda"
