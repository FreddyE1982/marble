"""Benchmark graph precompilation versus eager execution."""

from __future__ import annotations

import time

import torch

from graph_cache import GRAPH_CACHE
from marble_core import _simple_mlp, precompile_simple_mlp


def benchmark_precompile(repeats: int = 1000) -> dict[str, float]:
    """Return timing information for precompiled vs eager execution.

    Parameters
    ----------
    repeats:
        Number of forward passes to execute when timing.  A higher value
        increases accuracy of the measurement but also runtime.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.randn(1, 4, device=device)

    GRAPH_CACHE.enable(False)
    start = time.perf_counter()
    for _ in range(repeats):
        _simple_mlp(sample)
    no_pre = time.perf_counter() - start

    GRAPH_CACHE.clear()
    precompile_simple_mlp(sample)
    start = time.perf_counter()
    for _ in range(repeats):
        _simple_mlp(sample)
    pre = time.perf_counter() - start

    speedup = no_pre / pre if pre > 0 else float("inf")
    return {"no_precompile": no_pre, "precompiled": pre, "speedup": speedup}


if __name__ == "__main__":
    res = benchmark_precompile()
    print(
        f"no precompile: {res['no_precompile']:.4f}s, precompiled: {res['precompiled']:.4f}s, "
        f"speedup {res['speedup']:.2f}x"
    )
