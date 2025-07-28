import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config_loader import load_config

def test_extended_config_parameters():
    cfg = load_config()
    assert cfg["core"]["random_seed"] == 42
    assert cfg["neuronenblitz"]["plasticity_modulation"] == 1.0
    assert cfg["brain"]["model_name"] == "marble_default"
    assert cfg["remote_client"]["auth_token"] is None
    assert cfg["remote_server"]["ssl_enabled"] is False
    assert cfg["remote_server"]["auth_token"] is None
    assert cfg["metrics_visualizer"]["refresh_rate"] == 1
    assert cfg["brain"]["super_evolution_mode"] is False
