"""Simple object memory pools for neurons and synapses."""

from __future__ import annotations

from collections import deque
from typing import Callable


class MemoryPool:
    """Generic object pool with reference counting."""

    def __init__(self, factory: Callable[[], object], max_size: int = 1000) -> None:
        self.factory = factory
        self.max_size = max_size
        self._free: deque[object] = deque()

    def allocate(self) -> object:
        if self._free:
            return self._free.popleft()
        return self.factory()

    def release(self, obj: object) -> None:
        if len(self._free) < self.max_size:
            self._free.append(obj)
