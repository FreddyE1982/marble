import torch
import pytest

from marble_activation_kernel import marble_activation


def test_cpu_marble_activation_forward_and_backward():
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = marble_activation(x, threshold=0.0, a=2.0, b=1.0, c=0.5)
    expected = torch.where(x > 0.0, 2.0 * x + 1.0, 0.5 * x)
    assert torch.allclose(y, expected)

    y.sum().backward()
    expected_grad = torch.tensor([0.5, 2.0, 2.0])
    assert torch.allclose(x.grad, expected_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_matches_cpu():
    x = torch.randn(1024, device="cuda")
    cpu = x.cpu()
    out_gpu = marble_activation(x, threshold=0.1, a=1.5, b=0.2, c=0.7)
    out_cpu = marble_activation(cpu, threshold=0.1, a=1.5, b=0.2, c=0.7)
    assert torch.allclose(out_gpu.cpu(), out_cpu, atol=1e-6)

    x.requires_grad = True
    y = marble_activation(x, threshold=0.1, a=1.5, b=0.2, c=0.7)
    y.sum().backward()
    assert x.grad is not None
