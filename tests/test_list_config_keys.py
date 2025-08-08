from scripts.list_config_keys import list_config_keys


def test_list_config_keys_contains_known_parameters():
    keys = list_config_keys("config.yaml")
    assert "dataset.source" in keys
    assert "core.backend" in keys
    assert "dream_reinforcement_learning.dream_cycle_duration" in keys
