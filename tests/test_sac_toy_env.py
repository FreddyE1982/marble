import torch
import pytest
from examples.sac_toy_env import SACGridEnv


def test_sac_grid_env_cpu():
    env = SACGridEnv(grid_size=3, max_steps=5, device=torch.device("cpu"))
    state = env.reset()
    assert state.device.type == "cpu"
    assert state.sum().item() == 1.0
    state, reward, done, _ = env.step(torch.tensor(1))
    assert not done
    assert reward.item() == pytest.approx(-0.1)
    state, reward, done, _ = env.step(torch.tensor(1))
    assert done
    assert reward.item() == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sac_grid_env_gpu():
    device = torch.device("cuda")
    env = SACGridEnv(grid_size=3, max_steps=5, device=device)
    state = env.reset()
    assert state.device.type == "cuda"
    state, reward, done, _ = env.step(torch.tensor(1, device=device))
    assert state.device.type == "cuda"
    assert reward.device.type == "cuda"
