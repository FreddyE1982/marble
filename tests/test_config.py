import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config_loader import load_config, create_marble_from_config
from marble_main import MARBLE


def test_load_config_defaults():
    cfg = load_config()
    assert 'core' in cfg
    assert cfg['core']['width'] == 30
    assert 'neuronenblitz' in cfg
    assert cfg['brain']['save_threshold'] == 0.05
    assert cfg['meta_controller']['history_length'] == 5
    assert cfg['neuromodulatory_system']['initial']['emotion'] == "neutral"


def test_create_marble_from_config():
    marble = create_marble_from_config()
    assert isinstance(marble, MARBLE)
    assert marble.brain.meta_controller.history_length == 5
