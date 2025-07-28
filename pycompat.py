"""Compatibility helpers for older Python versions (3.8+)."""

from __future__ import annotations

import sys
from typing import Any, Callable

if sys.version_info < (3, 9):  # pragma: no cover - runtime check
    from typing_extensions import Annotated  # noqa: F401


def removeprefix(s: str, prefix: str) -> str:
    """Return ``s`` without the specified ``prefix`` (Python <3.9)."""
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def cached(func: Callable[..., Any]) -> Callable[..., Any]:
    """Lightweight cached decorator for Python 3.8."""
    cache: dict[tuple, Any] = {}

    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper
