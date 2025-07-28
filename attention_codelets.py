"""Plugin interface for attention codelets."""

from __future__ import annotations

from typing import Any, Callable, List

_codelets: List[Callable[[Any], None]] = []


def register_codelet(func: Callable[[Any], None]) -> None:
    """Register an attention codelet callback."""
    _codelets.append(func)


def get_codelets() -> list[Callable[[Any], None]]:
    """Return all registered codelet callbacks."""
    return list(_codelets)
