import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_utils import JSONFormatter, configure_structured_logging


def test_configure_structured_logging(tmp_path):
    log_file = tmp_path / "log.jsonl"
    logger = configure_structured_logging(
        True,
        str(log_file),
        level="WARNING",
        datefmt="%Y",
        propagate=True,
    )
    logger.info("ignore")
    logger.warning("hello")
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["msg"] == "hello"
    assert obj["level"] == "WARNING"
    assert len(obj["time"]) == 4  # custom date format applied
    assert logger.propagate is True


def test_disable_structured_logging(tmp_path):
    log_file = tmp_path / "plain.log"
    logger = configure_structured_logging(
        False,
        str(log_file),
        level="WARNING",
        format="%(levelname)s:%(message)s",
    )
    logger.info("ignore")
    logger.warning("warn")
    text = log_file.read_text(encoding="utf-8").strip()
    assert text == "WARNING:warn"
    assert not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers)


def test_log_rotation(tmp_path):
    log_file = tmp_path / "rotate.log"
    logger = configure_structured_logging(
        False,
        str(log_file),
        rotate=True,
        max_bytes=50,
        backup_count=1,
    )
    for _ in range(20):
        logger.warning("x" * 10)
    assert log_file.exists()
    rotated = tmp_path / "rotate.log.1"
    assert rotated.exists()
