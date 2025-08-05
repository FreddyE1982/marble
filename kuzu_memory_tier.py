from __future__ import annotations

"""Persistent hybrid memory storage backed by a Kùzu graph database."""

from typing import Any, List, Tuple
from datetime import datetime, UTC

import numpy as np

from kuzu_interface import KuzuGraphDatabase


class KuzuMemoryTier:
    """Store key/value pairs and embedding vectors in a Kùzu database.

    Each memory entry is stored as a node in the database with properties for
    the key, associated value, embedding vector and insertion timestamp. The
    class offers similarity search by loading all vectors into memory and
    computing cosine similarity in numpy. ``forget_old`` removes entries beyond
    a specified limit based on their timestamp."""

    def __init__(self, db_path: str, dim: int) -> None:
        self.db = KuzuGraphDatabase(db_path)
        self.dim = dim
        self._init_schema()

    # ------------------------------------------------------------------
    # schema management
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        """Ensure the ``Memory`` node table exists."""
        try:
            self.db.create_node_table(
                "Memory",
                {
                    "key": "STRING",
                    "value": "DOUBLE",
                    "vector": "DOUBLE[]",
                    "timestamp": "STRING",
                },
                "key",
            )
        except Exception:
            # table already present
            pass

    # ------------------------------------------------------------------
    # data manipulation
    # ------------------------------------------------------------------
    def add(self, key: Any, value: float, vector: np.ndarray) -> None:
        """Insert or update a memory entry."""
        if vector.shape[0] != self.dim:
            vec = vector.reshape(-1)[: self.dim]
            if vec.shape[0] < self.dim:
                vec = np.pad(vec, (0, self.dim - vec.shape[0]))
            vector = vec
        params = {
            "key": key,
            "value": float(value),
            "vector": vector.astype(float).tolist(),
            "ts": datetime.now(UTC).isoformat(),
        }
        self.db.execute(
            "MERGE (m:Memory {key: $key}) "
            "SET m.value = $value, m.vector = $vector, m.timestamp = $ts;",
            params,
        )

    def retrieve(self, key: Any) -> Any:
        """Return the value stored for ``key`` or ``None``."""
        rows = self.db.execute(
            "MATCH (m:Memory {key: $key}) RETURN m.value AS value;", {"key": key}
        )
        if rows:
            return rows[0]["value"]
        return None

    def query(self, vector: np.ndarray, top_k: int = 3) -> List[Tuple[Any, Any]]:
        """Return ``top_k`` keys with highest cosine similarity to ``vector``."""
        rows = self.db.execute(
            "MATCH (m:Memory) RETURN m.key AS key, m.value AS value, m.vector AS vector;"
        )
        if not rows:
            return []
        vec = vector.astype(float).reshape(1, -1)
        mat = np.array([r["vector"] for r in rows], dtype=float)
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
        denom = np.where(denom == 0, 1.0, denom)
        sims = (mat @ vec.T).flatten() / denom
        idx = np.argsort(sims)[::-1][:top_k]
        return [(rows[i]["key"], rows[i]["value"]) for i in idx]

    def forget_old(self, max_entries: int) -> None:
        """Trim oldest entries to keep at most ``max_entries`` records."""
        rows = self.db.execute(
            "MATCH (m:Memory) RETURN m.key AS key ORDER BY m.timestamp;"
        )
        if len(rows) <= max_entries:
            return
        excess = rows[:-max_entries]
        for r in excess:
            self.db.delete_node("Memory", "key", r["key"])


__all__ = ["KuzuMemoryTier"]
