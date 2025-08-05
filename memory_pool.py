"""Simple object memory pools for neurons and synapses."""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Callable, Iterator, Tuple

import numpy as np

try:  # optional dependency for GPU tensor pooling
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover - jax not installed
    jnp = None

try:  # optional dependency for torch tensors
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None


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


class ArrayMemoryPool(MemoryPool):
    """Memory pool specialised for reusable array/tensor buffers.

    Parameters
    ----------
    shape:
        Shape of arrays managed by the pool.
    dtype:
        Data type for allocated arrays.  Expects ``numpy`` dtype for NumPy/JAX
        pools and ``torch`` dtype for PyTorch pools.
    backend:
        ``"numpy"`` (default), ``"jax"`` or ``"torch"`` to control the
        allocation framework.  ``"jax"`` pools allocate device arrays which may
        reside on CPU or GPU depending on the active JAX device.  ``"torch``
        pools allocate tensors on CPU or GPU depending on ``device`` or
        availability of CUDA.
    device:
        Optional device specifier for ``torch`` or ``jax`` pools.  If ``None``,
        ``torch`` pools default to ``"cuda"`` when available and ``jax`` uses
        its current default device.
    max_size:
        Maximum number of buffers held in the pool.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: object | None = None,
        *,
        backend: str = "numpy",
        device: str | None = None,
        max_size: int = 1000,
    ) -> None:
        self.shape = shape
        self.backend = backend
        self.dtype = dtype or (torch.float32 if backend == "torch" else np.float32)
        self.device = device

        def factory() -> object:
            if backend == "numpy":
                return np.empty(shape, dtype=self.dtype)
            if backend == "jax":
                if jnp is None:  # pragma: no cover - safety
                    raise RuntimeError("JAX is not available")
                return jnp.empty(shape, dtype=self.dtype, device=device)
            if backend == "torch":
                if torch is None:  # pragma: no cover - safety
                    raise RuntimeError("PyTorch is not available")
                dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
                return torch.empty(shape, dtype=self.dtype, device=dev)
            raise ValueError(f"Unknown backend: {backend}")

        super().__init__(factory, max_size=max_size)
