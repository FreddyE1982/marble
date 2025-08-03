"""Tensor backend abstraction supporting NumPy and JAX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # optional dependency
    import jax.numpy as jnp  # type: ignore
    from jax import device_get  # type: ignore
    _HAS_JAX = True
except Exception:  # pragma: no cover - jax not installed
    jnp = None
    device_get = lambda x: x  # type: ignore
    _HAS_JAX = False


@dataclass
class _Backend:
    """Abstract backend defining basic tensor ops."""

    def matmul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def sigmoid(self, x: Any) -> Any:
        raise NotImplementedError

    def relu(self, x: Any) -> Any:
        raise NotImplementedError


class _NumpyBackend(_Backend):
    def matmul(self, a: Any, b: Any) -> Any:  # pragma: no cover - trivial
        return np.matmul(a, b)

    def sigmoid(self, x: Any) -> Any:  # pragma: no cover - trivial
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x: Any) -> Any:  # pragma: no cover - trivial
        return np.maximum(x, 0)


class _JaxBackend(_Backend):
    def matmul(self, a: Any, b: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return device_get(jnp.matmul(jnp.asarray(a), jnp.asarray(b)))

    def sigmoid(self, x: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return device_get(1.0 / (1.0 + jnp.exp(-jnp.asarray(x))))

    def relu(self, x: Any) -> Any:
        if jnp is None:  # pragma: no cover - safety
            raise RuntimeError("JAX is not available")
        return device_get(jnp.maximum(jnp.asarray(x), 0))


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


def matmul(a: Any, b: Any) -> Any:
    """Return matrix product of ``a`` and ``b`` using active backend."""
    return _backend.matmul(a, b)


def sigmoid(x: Any) -> Any:
    """Return elementwise sigmoid of ``x`` using active backend."""
    return _backend.sigmoid(x)


def relu(x: Any) -> Any:
    """Return elementwise ReLU of ``x`` using active backend."""
    return _backend.relu(x)
