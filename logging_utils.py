import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Union


class JSONFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def __init__(self, datefmt: Optional[str] = None):
        super().__init__(datefmt=datefmt)

    def format(
        self, record: logging.LogRecord
    ) -> str:  # pragma: no cover - simple formatting
        data = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_structured_logging(
    enabled: bool,
    log_file: Optional[str] = None,
    *,
    level: Union[int, str] = "INFO",
    format: str = "%(levelname)s:%(name)s:%(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    propagate: bool = False,
    rotate: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    encoding: str = "utf-8",
) -> logging.Logger:
    """Configure root logger.

    Parameters
    ----------
    enabled:
        When ``True`` logs are formatted as JSON objects. Otherwise standard logging
        configuration is used.
    log_file:
        Optional path to a file where logs are written. When ``None`` logs are
        output to ``stderr``.
    level:
        Minimum severity of events to record. Accepts either an integer or one of
        the string level names defined by :mod:`logging` (e.g. ``"DEBUG"``).
    format:
        Format string used for non-structured logging. Ignored when ``enabled`` is
        ``True``.
    datefmt:
        ``strftime``-compatible format string for timestamps.
    propagate:
        Forward log records to ancestor loggers when ``True``.
    rotate:
        When ``True`` a :class:`~logging.handlers.RotatingFileHandler` is used for
        ``log_file`` with the specified ``max_bytes`` and ``backup_count``.
    max_bytes:
        Maximum size in bytes of a log file before rotation occurs.
    backup_count:
        Number of rotated log files to keep.
    encoding:
        Text encoding used for file-based handlers.
    """

    logger = logging.getLogger()
    logger.propagate = propagate
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if isinstance(level, str):
        level_value = logging._nameToLevel.get(level.upper(), logging.INFO)
    else:
        level_value = level

    if log_file:
        if rotate:
            handler: logging.Handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=encoding,
            )
        else:
            handler = logging.FileHandler(log_file, encoding=encoding)
    else:
        handler = logging.StreamHandler()

    if enabled:
        handler.setFormatter(JSONFormatter(datefmt=datefmt))
    else:
        handler.setFormatter(logging.Formatter(format, datefmt))

    logger.addHandler(handler)
    logger.setLevel(level_value)
    return logger
