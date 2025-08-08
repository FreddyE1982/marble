from scripts.list_config_keys import find_unused_keys, list_config_keys

import yaml


def test_list_config_keys_contains_known_parameters():
    keys = list_config_keys("config.yaml")
    assert "dataset.source" in keys
    assert "core.backend" in keys
    assert "dream_reinforcement_learning.dream_cycle_duration" in keys


def test_find_unused_keys_detects_unreferenced_parameters(tmp_path):
    """Ensure configuration keys absent from code are flagged as unused."""

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"alpha": 1, "beta": 2}), encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    (src / "example.py").write_text("value = config['alpha']\n", encoding="utf-8")

    unused = find_unused_keys(cfg, root=src)
    assert "beta" in unused
    assert "alpha" not in unused
