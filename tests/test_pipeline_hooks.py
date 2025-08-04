import gc
import torch
import pytest

from pipeline import Pipeline


def double_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2


def test_hook_registration_and_removal():
    pipe = Pipeline()
    pipe.add_step(
        "double_tensor",
        module="tests.test_pipeline_hooks",
        params={"tensor": torch.tensor([1, 2])},
        name="double",
    )
    calls: list[str] = []

    def pre(step, marble, device):
        calls.append("pre")

    pipe.register_pre_hook("double", pre)
    pipe.remove_pre_hook("double", pre)
    pipe.execute()
    assert calls == []


def test_pre_post_hooks_cpu():
    pipe = Pipeline()
    tensor = torch.tensor([1, 2])
    pipe.add_step(
        "double_tensor",
        module="tests.test_pipeline_hooks",
        params={"tensor": tensor},
        name="double",
    )

    def pre(step, marble, device):
        step["params"]["tensor"] = step["params"]["tensor"].to(device) * 3

    def post(step, result, marble, device):
        assert result.device.type == device.type
        return result + 1

    pipe.register_pre_hook("double", pre)
    pipe.register_post_hook("double", post)
    result = pipe.execute()[0]
    expected = torch.tensor([7, 13])
    assert torch.equal(result.cpu(), expected)


def test_pre_post_hooks_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    pipe = Pipeline()
    tensor = torch.tensor([1, 2], device=device)
    pipe.add_step(
        "double_tensor",
        module="tests.test_pipeline_hooks",
        params={"tensor": tensor},
        name="double",
    )

    def pre(step, marble, dev):
        assert dev.type == "cuda"
        step["params"]["tensor"] = step["params"]["tensor"] * 3

    def post(step, result, marble, dev):
        assert result.device.type == "cuda"
        return result + 1

    pipe.register_pre_hook("double", pre)
    pipe.register_post_hook("double", post)

    before = torch.cuda.memory_allocated()
    result = pipe.execute()[0]
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert torch.all(result == torch.tensor([7, 13], device=device))
    assert after <= before + 1024
    # clear any lingering refs
    del result
    gc.collect()
    torch.cuda.empty_cache()
