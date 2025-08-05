import os
import yaml
import pytest

from config_editor import load_config_text, save_config_text


def test_save_config_creates_backup_and_updates(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("core:\n  representation_size: 4\n")
    new_yaml = "core:\n  representation_size: 8\n"
    backup_path = save_config_text(new_yaml, path=str(cfg))
    assert yaml.safe_load(cfg.read_text())["core"]["representation_size"] == 8
    assert os.path.exists(backup_path)


def test_save_config_invalid_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("core:\n  representation_size: 4\n")
    with pytest.raises(Exception):
        save_config_text("core: [unclosed", path=str(cfg))


def test_load_config_text(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("core:\n  representation_size: 4\n")
    assert "representation_size" in load_config_text(path=str(cfg))
