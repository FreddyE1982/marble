"""Simple object memory pools for neurons and synapses."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterator
import contextlib


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

    def preallocate(self, count: int) -> None:
        """Pre-fill the pool with ``count`` new objects."""
        for _ in range(count):
            if len(self._free) >= self.max_size:
                break
            self._free.append(self.factory())

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of available objects in the pool."""
        return len(self._free)

    @contextlib.contextmanager
    def borrow(self) -> Iterator[object]:
        """Context manager yielding an allocated object and releasing it on exit."""
        obj = self.allocate()
        try:
            yield obj
        finally:
            self.release(obj)
