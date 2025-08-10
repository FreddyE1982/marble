import os
import yaml
from config_loader import create_marble_from_config


def _write_cfg(tmp_path, cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def test_reinforcement_learning_section(tmp_path):
    cfg = {
        "reinforcement_learning": {
            "enabled": True,
            "episodes": 1,
            "max_steps": 1,
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)
    marble = create_marble_from_config(cfg_path)
    assert hasattr(marble, "rl_agent")


def test_quantum_flux_learning_section(tmp_path):
    cfg = {
        "quantum_flux_learning": {
            "enabled": True,
            "epochs": 1,
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)
    marble = create_marble_from_config(cfg_path)
    assert hasattr(marble, "quantum_flux_learner")
