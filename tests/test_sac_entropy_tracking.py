import pytest
import torch

from marble_core import Core
from marble_neuronenblitz import (
    Neuronenblitz,
    enable_sac,
    sac_update,
    plot_sac_entropy,
)


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_entropy_history_and_plot(tmp_path, device: str) -> None:
    core = Core({"width": 1, "height": 1})
    nb = Neuronenblitz(core)
    enable_sac(nb, state_dim=3, action_dim=2, device=device)
    state = torch.randn(3, device=device)
    action = torch.randn(2, device=device)
    next_state = torch.randn(3, device=device)
    sac_update(nb, state, action, reward=1.0, next_state=next_state, done=False)
    assert len(nb.sac_entropy_history) == 1
    out_file = tmp_path / "entropy.png"
    plot_sac_entropy(nb, out_file)
    assert out_file.exists() and out_file.stat().st_size > 0
