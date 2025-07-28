"""Episodic Memory plugin structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EpisodicEntry:
    """Single episode snapshot used by planning modules."""

    context: Dict[str, Any]
    reward: float
    outcome: Any
    timestamp: float

