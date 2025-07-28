"""Episodic Memory plugin structures and utilities."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Sequence


@dataclass
class EpisodicEntry:
    """Single episode snapshot used by planning modules."""

    context: Dict[str, Any]
    reward: float
    outcome: Any
    timestamp: float = field(default_factory=time)


class EpisodicMemory:
    """Maintain transient and long‑term episodic memory."""

    def __init__(
        self,
        *,
        transient_capacity: int = 50,
        storage_path: str | None = None,
    ) -> None:
        self.transient: Deque[EpisodicEntry] = deque(maxlen=transient_capacity)
        self.long_term: List[EpisodicEntry] = []
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path and self.storage_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # persistence utilities
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text())
            self.long_term = [EpisodicEntry(**e) for e in data]
        except Exception:
            self.long_term = []

    def _save(self) -> None:
        if not self.storage_path:
            return
        data = [e.__dict__ for e in self.long_term]
        self.storage_path.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add_episode(self, context: Dict[str, Any], reward: float, outcome: Any) -> None:
        """Store an episode in memory."""

        entry = EpisodicEntry(dict(context), float(reward), outcome)
        self.transient.append(entry)
        self.long_term.append(entry)
        self._save()

    def query(
        self,
        context: Dict[str, Any],
        *,
        k: int = 1,
        similarity_fn: Callable[[Dict[str, Any], Dict[str, Any]], float] | None = None,
    ) -> List[EpisodicEntry]:
        """Return ``k`` most similar episodes by context."""

        if similarity_fn is None:
            similarity_fn = self._default_similarity
        episodes = list(self.long_term)
        scored = [(similarity_fn(context, e.context), e) for e in episodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    # ------------------------------------------------------------------
    def _default_similarity(
        self, ctx_a: Dict[str, Any], ctx_b: Dict[str, Any]
    ) -> float:
        match = sum(1 for k in ctx_a if k in ctx_b and ctx_a[k] == ctx_b[k])
        return match / (len(ctx_a) + len(ctx_b) - match + 1e-9)


def register(*_: Callable[[str], None]) -> None:
    """Plugin entry point for the loader."""
    return
