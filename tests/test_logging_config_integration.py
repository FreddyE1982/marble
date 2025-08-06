import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_loader import create_marble_from_config


def test_logging_config_applied(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    log_file = tmp_path / "log.jsonl"
    cfg_path.write_text(f"logging:\n  structured: true\n  log_file: '{log_file}'\n")
    create_marble_from_config(str(cfg_path))
    logging.getLogger().info("hello")
    text = log_file.read_text(encoding="utf-8").strip()
    obj = json.loads(text)
    assert obj["msg"] == "hello"
    assert obj["level"] == "INFO"
