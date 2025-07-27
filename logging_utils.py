import json
import logging
from typing import Optional

class JSONFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple formatting
        data = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_structured_logging(enabled: bool, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger for structured logging.

    Parameters
    ----------
    enabled:
        When ``True`` logs are formatted as JSON objects. Otherwise standard logging
        configuration is used.
    log_file:
        Optional path to a file where logs are written. When ``None`` logs are
        output to ``stderr``.
    """
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    if not enabled:
        logging.basicConfig(level=logging.INFO)
        return logger

    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
