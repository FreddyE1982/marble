"""Tensor backend abstraction supporting NumPy and JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from memory_pool import MemoryPool

try:  # optional dependency
    import jax.numpy as jnp  # type: ignore
    from jax import device_get  # type: ignore
    from jax import random as jrandom  # type: ignore

    _HAS_JAX = True
except Exception:  # pragma: no cover - jax not installed
    jnp = None

    def device_get(x):  # type: ignore  # noqa: D401
        """Identity fallback when JAX is unavailable."""
        return x

    _HAS_JAX = False

try:  # optional torch dependency for GPU tensor support
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None


@dataclass
class _Backend:
    """Abstract backend defining basic tensor ops."""

    def matmul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def sigmoid(self, x: Any) -> Any:
        raise NotImplementedError

    def relu(self, x: Any) -> Any:
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def rand(self, shape: Tuple[int, ...]) -> Any:
        raise NotImplementedError

    def randn(self, shape: Tuple[int, ...]) -> Any:
        raise NotImplementedError


class _NumpyBackend(_Backend):
    def __init__(self) -> None:  # pragma: no cover - simple init
        self.rng = np.random.default_rng()

    def matmul(self, a: Any, b: Any) -> Any:  # pragma: no cover - trivial
        return np.matmul(a, b)

    def sigmoid(self, x: Any) -> Any:  # pragma: no cover - trivial
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x: Any) -> Any:  # pragma: no cover - trivial
        return np.maximum(x, 0)

    def seed(self, seed: int) -> None:  # pragma: no cover - trivial
        self.rng = np.random.default_rng(seed)

    def rand(self, shape: Tuple[int, ...]) -> Any:  # pragma: no cover - trivial
        return self.rng.random(shape)

    def randn(self, shape: Tuple[int, ...]) -> Any:  # pragma: no cover - trivial
        return self.rng.standard_normal(shape)


class _JaxBackend(_Backend):
    def __init__(self) -> None:  # pragma: no cover - simple init
        if jnp is None:
            raise RuntimeError("JAX is not available")
        self.key = jrandom.PRNGKey(0)

    def _split(self) -> "jnp.ndarray":  # pragma: no cover - helper
        self.key, sub = jrandom.split(self.key)
        return sub

    def matmul(self, a: Any, b: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return jnp.matmul(jnp.asarray(a), jnp.asarray(b))

    def sigmoid(self, x: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return 1.0 / (1.0 + jnp.exp(-jnp.asarray(x)))

    def relu(self, x: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return jnp.maximum(jnp.asarray(x), 0)

    def seed(self, seed: int) -> None:  # pragma: no cover - simple
        self.key = jrandom.PRNGKey(seed)

    def rand(self, shape: Tuple[int, ...]) -> Any:
        sub = self._split()
        return jrandom.uniform(sub, shape)

    def randn(self, shape: Tuple[int, ...]) -> Any:
        sub = self._split()
        return jrandom.normal(sub, shape)


_backend: _Backend = _NumpyBackend()


def set_backend(name: str) -> None:
    """Select tensor backend by ``name``.

    Parameters
    ----------
    name:
        ``"numpy"`` (default) or ``"jax"``. ``"jax"`` requires the ``jax``
        package to be installed.
    """

    global _backend
    name = name.lower()
    if name == "numpy":
        _backend = _NumpyBackend()
    elif name == "jax":
        if not _HAS_JAX:
            raise ImportError("JAX backend requested but jax is not installed")
        _backend = _JaxBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")


def seed(seed: int) -> None:
    """Seed the random number generator of the active backend."""
    _backend.seed(int(seed))


def rand(shape: Tuple[int, ...]) -> Any:
    """Return uniform random array with given ``shape``."""
    return _backend.rand(shape)


def randn(shape: Tuple[int, ...]) -> Any:
    """Return standard normal random array with given ``shape``."""
    return _backend.randn(shape)


def xp():
    """Return array module (numpy or jax) of active backend."""
    return np if isinstance(_backend, _NumpyBackend) else jnp


def to_numpy(arr: Any) -> np.ndarray:
    """Convert ``arr`` from backend format to a NumPy array."""
    if torch is not None and isinstance(arr, torch.Tensor):  # pragma: no cover - torch
        return arr.detach().cpu().numpy()
    if jnp is not None and isinstance(arr, jnp.ndarray):  # pragma: no cover - jax
        return np.asarray(device_get(arr))
    return np.asarray(arr)


def matmul(a: Any, b: Any, *, out_pool: "MemoryPool" | None = None) -> Any:
    """Return matrix product of ``a`` and ``b`` using active backend.

    Parameters
    ----------
    a, b:
        Input arrays or tensors.
    out_pool:
        Optional :class:`~memory_pool.MemoryPool` providing preallocated output
        buffers.  When supplied the result is written into a buffer from the
        pool, allowing callers to manage memory reuse explicitly.  The buffer
        must match the shape and dtype of the operation result.
    """

    result = _backend.matmul(a, b)
    if out_pool is None:
        return result
    out = out_pool.allocate()
    _copy_to(result, out)
    return out


def sigmoid(x: Any, *, out_pool: "MemoryPool" | None = None) -> Any:
    """Return elementwise sigmoid of ``x`` using active backend."""
    result = _backend.sigmoid(x)
    if out_pool is None:
        return result
    out = out_pool.allocate()
    _copy_to(result, out)
    return out


def relu(x: Any, *, out_pool: "MemoryPool" | None = None) -> Any:
    """Return elementwise ReLU of ``x`` using active backend."""
    result = _backend.relu(x)
    if out_pool is None:
        return result
    out = out_pool.allocate()
    _copy_to(result, out)
    return out


def _copy_to(src: Any, dst: Any) -> None:
    """Best-effort copy from ``src`` to ``dst`` supporting NumPy, JAX and torch."""
    if torch is not None and hasattr(torch, "Tensor") and isinstance(dst, torch.Tensor):  # type: ignore
        if isinstance(src, torch.Tensor):
            dst.copy_(src)
        else:
            dst.copy_(torch.as_tensor(src, dtype=dst.dtype, device=dst.device))
        return
    if (
        jnp is not None and hasattr(jnp, "ndarray") and isinstance(dst, jnp.ndarray)
    ):  # pragma: no cover - rarely used
        dst = dst.at[:].set(jnp.asarray(src))
        return
    np.copyto(dst, src)
