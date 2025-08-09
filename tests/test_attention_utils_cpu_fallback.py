import pytest
import torch

from attention_utils import GatingLayer, generate_causal_mask

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
