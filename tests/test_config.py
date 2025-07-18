from config_loader import load_config


def test_load_config_defaults():
    cfg = load_config()
    assert 'core' in cfg
    assert cfg['core']['width'] == 30
    assert 'neuronenblitz' in cfg
    assert cfg['brain']['save_threshold'] == 0.05
