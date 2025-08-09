import pytest
import torch

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.learning import enable_sac, sac_select_action, sac_update


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_sac_select_action_and_update(device: str) -> None:
    core = Core({"width": 1, "height": 1})
    nb = Neuronenblitz(core)
    enable_sac(nb, state_dim=3, action_dim=2, device=device)
    state = torch.randn(3, device=device)
    action, log_prob = sac_select_action(nb, state)
    assert action.shape == (2,)
    assert log_prob.shape == (1,)
    next_state = torch.randn(3, device=device)
    # capture critic params before update
    before = [p.clone() for p in nb.sac_critic.parameters()]
    sac_update(nb, state, action, reward=1.0, next_state=next_state, done=False)
    after = list(nb.sac_critic.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
