import pytest
import torch

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.learning import enable_sac


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.mark.parametrize("device", devices)
def test_enable_sac_instantiates_networks(device: str) -> None:
    core = Core({"width": 1, "height": 1})
    nb = Neuronenblitz(core)
    enable_sac(nb, state_dim=3, action_dim=2, device=device)
    assert nb.sac_actor.device.type == device
    state = torch.randn(1, 3, device=nb.sac_actor.device)
    action, log_prob = nb.sac_actor(state)
    q1, q2 = nb.sac_critic(state, action)
    assert action.shape == (1, 2)
    assert log_prob.shape == (1, 1)
    assert q1.shape == (1, 1)
    assert q2.shape == (1, 1)


def test_enable_sac_reads_temperature_from_config(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("sac:\n  temperature: 0.05\n")
    import config_loader

    monkeypatch.setattr(config_loader, "DEFAULT_CONFIG_FILE", cfg_path)

    core = Core({"width": 1, "height": 1})
    nb = Neuronenblitz(core)
    enable_sac(nb, state_dim=3, action_dim=2, tune_entropy=False)
    assert float(nb.sac_alpha) == pytest.approx(0.05)
