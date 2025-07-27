import os, sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from logging_utils import configure_structured_logging, JSONFormatter


def test_configure_structured_logging(tmp_path):
    log_file = tmp_path / "log.jsonl"
    logger = configure_structured_logging(True, str(log_file))
    logger.info("hello")
    with open(log_file, "r", encoding="utf-8") as f:
        data = f.read().strip()
    assert data
    obj = json.loads(data)
    assert obj["msg"] == "hello"
    assert obj["level"] == "INFO"


def test_disable_structured_logging():
    logger = configure_structured_logging(False)
    assert not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)
