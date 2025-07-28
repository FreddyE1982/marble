"""Enhanced context history entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ContextEntry:
    """Single context history record."""

    context: Dict[str, Any]
    markers: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    theory_of_mind: Dict[str, Any] = field(default_factory=dict)
