import pytest
import torch

from attention_utils import (
    GatingLayer,
    benchmark_mask_overhead,
    gate_figure,
    generate_causal_mask,
    mask_figure,
)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_generate_causal_mask_parity(device: str) -> None:
    mask = generate_causal_mask(4, device=torch.device(device))
    assert mask.device.type == device
    cpu_mask = generate_causal_mask(4)
    assert torch.equal(mask.cpu(), cpu_mask)


@pytest.mark.parametrize("mode", ["sine", "chaos"])
@pytest.mark.parametrize("device", devices)
def test_gating_layer_parity(device: str, mode: str) -> None:
    layer = GatingLayer(mode=mode).to(device)
    gate = layer(6, device=torch.device(device))
    assert gate.device.type == device
    cpu_gate = GatingLayer(mode=mode)(6)
    assert torch.allclose(gate.cpu(), cpu_gate, atol=1e-6)


@pytest.mark.parametrize("device", devices)
def test_benchmark_mask_overhead_runs(device: str) -> None:
    """Ensure benchmark executes on both CPU and GPU and returns non-negative time."""
    t = benchmark_mask_overhead(4, device=device, repeats=2)
    assert isinstance(t, float)
    assert t >= 0.0


@pytest.mark.parametrize("device", devices)
def test_mask_figure_accepts_tensor(device: str) -> None:
    """mask_figure should handle tensors from different devices."""
    mask = generate_causal_mask(3, device=torch.device(device))
    fig = mask_figure(mask)
    assert fig.data[0].z.shape == (3, 3)


@pytest.mark.parametrize("mode", ["sine", "chaos"])
@pytest.mark.parametrize("device", devices)
def test_gate_figure_accepts_tensor(device: str, mode: str) -> None:
    """gate_figure should render gating values regardless of device."""
    layer = GatingLayer(mode=mode).to(device)
    gate = layer(5, device=torch.device(device))
    fig = gate_figure(gate)
    assert len(fig.data[0].y) == 5
