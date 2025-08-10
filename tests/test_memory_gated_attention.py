import torch
import pytest

from attention_utils import GatingLayer
from episodic_memory import EpisodicMemory

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_memory_gating_uses_reward(device: str) -> None:
    memory = EpisodicMemory()
    memory.add_episode({"task": "x"}, reward=2.0, outcome=None)
    layer = GatingLayer(mode="memory").to(device)
    gate = layer(
        4,
        device=torch.device(device),
        memory=memory,
        context={"task": "x"},
    )
    expected = torch.full(
        (4,), torch.tanh(torch.tensor(2.0)), device=device, dtype=torch.float32
    )
    assert torch.allclose(gate, expected)
