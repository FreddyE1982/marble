"""Utilities for precompiling and caching compute graphs.

This module provides a lightweight cache for Torch compute graphs. When
precompilation is enabled a callable can be traced or scripted for a
specific input signature and reused across training iterations.  This
avoids repeated graph construction and yields small but measurable speed
ups on stable model structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Tuple

import torch


@dataclass(frozen=True)
class GraphKey:
    """Key describing a compiled graph.

    The key captures the input shape, dtype and device in addition to an
    arbitrary name.  The name allows multiple callables with the same
    input signature to coexist in the cache.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    extras: Tuple[Hashable, ...] = ()


class GraphCache:
    """Cache of ``torch.jit`` compiled callables."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self._cache: Dict[GraphKey, Callable[[torch.Tensor], torch.Tensor]] = {}

    # public API -------------------------------------------------
    def enable(self, flag: bool) -> None:
        """Enable or disable graph caching globally."""

        self.enabled = flag

    def clear(self) -> None:
        """Remove all cached graphs."""

        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)

    def precompile(
        self,
        key: GraphKey,
        fn: Callable[[torch.Tensor], torch.Tensor],
        example: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a compiled version of ``fn`` for ``example``.

        When caching is disabled the original ``fn`` is returned.  When
        enabled the function is compiled with ``torch.jit.trace`` on the
        provided example tensor and stored under ``key`` for reuse.
        """

        if not self.enabled:
            return fn
        if key not in self._cache:
            self._cache[key] = torch.jit.trace(fn, example)
        return self._cache[key]


# Global cache instance used throughout the project
GRAPH_CACHE = GraphCache()

__all__ = ["GRAPH_CACHE", "GraphCache", "GraphKey"]
