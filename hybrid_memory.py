import os
import pickle
from datetime import datetime, UTC
from typing import Any, List, Tuple

import numpy as np

from marble_core import perform_message_passing
from marble_neuronenblitz import Neuronenblitz


class VectorStore:
    """Store embeddings for similarity-based retrieval."""

    def __init__(self, path: str, dim: int) -> None:
        self.path = path
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.keys: List[Any] = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                    self.vectors, self.keys = data
            except Exception:
                self.vectors, self.keys = [], []

    def _save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump((self.vectors, self.keys), f)

    def add(self, key: Any, vector: np.ndarray) -> None:
        if vector.shape[0] != self.dim:
            vector = vector.reshape(-1)[: self.dim]
            if vector.shape[0] < self.dim:
                pad = np.zeros(self.dim - vector.shape[0])
                vector = np.concatenate([vector, pad])
        self.vectors.append(vector.astype(float))
        self.keys.append(key)
        self._save()

    def query(self, vector: np.ndarray, top_k: int = 3) -> List[Any]:
        if not self.vectors:
            return []
        vec = vector.astype(float).reshape(1, -1)
        mat = np.stack(self.vectors)
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
        denom = np.where(denom == 0, 1.0, denom)
        sims = (mat @ vec.T).flatten() / denom
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.keys[i] for i in idx]


class SymbolicMemory:
    """Dictionary-based persistent memory."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}

    def _save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)

    def store(self, key: Any, value: Any) -> None:
        self.data[key] = {"value": value, "timestamp": datetime.now(UTC)}
        self._save()

    def retrieve(self, key: Any) -> Any:
        item = self.data.get(key)
        if item is None:
            return None
        return item["value"]


class HybridMemory:
    """Hybrid memory using vector and symbolic stores."""

    def __init__(
        self,
        core,
        neuronenblitz: Neuronenblitz,
        vector_path: str = "vector_store.pkl",
        symbolic_path: str = "symbolic_memory.pkl",
    ) -> None:
        self.core = core
        self.nb = neuronenblitz
        self.vector_store = VectorStore(vector_path, core.rep_size)
        self.symbolic_memory = SymbolicMemory(symbolic_path)

    def _embed(self, value: float) -> np.ndarray:
        self.nb.dynamic_wander(value, apply_plasticity=False)
        perform_message_passing(self.core)
        rep_matrix = np.stack([n.representation for n in self.core.neurons])
        return rep_matrix.mean(axis=0)

    def store(self, key: Any, value: float) -> None:
        vec = self._embed(value)
        self.vector_store.add(key, vec)
        self.symbolic_memory.store(key, value)

    def retrieve(self, query: float, top_k: int = 3) -> List[Tuple[Any, Any]]:
        q_vec = self._embed(query)
        keys = self.vector_store.query(q_vec, top_k=top_k)
        return [(k, self.symbolic_memory.retrieve(k)) for k in keys]

    def forget_old(self, max_entries: int = 1000) -> None:
        if len(self.vector_store.keys) <= max_entries:
            return
        self.vector_store.keys = self.vector_store.keys[-max_entries:]
        self.vector_store.vectors = self.vector_store.vectors[-max_entries:]
        self.vector_store._save()
        excess = list(self.symbolic_memory.data.keys())[:-max_entries]
        for k in excess:
            self.symbolic_memory.data.pop(k, None)
        self.symbolic_memory._save()
