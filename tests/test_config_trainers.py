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


def test_semi_supervised_learning_section(tmp_path):
    cfg = {
        "semi_supervised_learning": {
            "enabled": True,
            "epochs": 1,
            "batch_size": 1,
            "unlabeled_weight": 0.5,
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)
    marble = create_marble_from_config(cfg_path)
    assert hasattr(marble, "semi_supervised_learner")


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


def test_synaptic_echo_learning_section(tmp_path):
    cfg = {
        "synaptic_echo_learning": {
            "enabled": True,
            "epochs": 1,
            "echo_influence": 1.0,
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)
    marble = create_marble_from_config(cfg_path)
    assert hasattr(marble, "synaptic_echo_learner")


def test_fractal_dimension_learning_section(tmp_path):
    cfg = {
        "fractal_dimension_learning": {
            "enabled": True,
            "epochs": 1,
            "target_dimension": 1.0,
        }
    }
    cfg_path = _write_cfg(tmp_path, cfg)
    marble = create_marble_from_config(cfg_path)
    assert hasattr(marble, "fractal_dimension_learner")
