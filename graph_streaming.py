"""Utilities for streaming large graphs in memory-constrained environments."""

from __future__ import annotations

from typing import Iterator, Tuple

from marble_core import Core


def stream_graph_chunks(core: Core, chunk_size: int) -> Iterator[Tuple[list, list]]:
    """Yield chunks of neurons and synapses from ``core``.

    Parameters
    ----------
    core:
        The :class:`Core` instance.
    chunk_size:
        Number of neurons per chunk.
    """
    n = len(core.neurons)
    for start in range(0, n, chunk_size):
        neuron_slice = core.neurons[start : start + chunk_size]
        syn_slice = [
            s
            for s in core.synapses
            if start <= s.source < start + chunk_size
            or start <= s.target < start + chunk_size
        ]
        yield neuron_slice, syn_slice


def identify_memory_hotspots(core: Core) -> dict[str, int]:
    """Return a mapping of structure name to approximate memory usage in bytes."""
    neuron_bytes = len(core.neurons) * core.rep_size * 8
    synapse_bytes = len(core.synapses) * 64
    return {"neurons": neuron_bytes, "synapses": synapse_bytes}


def benchmark_streaming(core: Core, chunk_size: int, iters: int = 10) -> float:
    """Benchmark ``stream_graph_chunks`` and return chunks per second."""
    import time

    start = time.perf_counter()
    for _ in range(iters):
        for _ in stream_graph_chunks(core, chunk_size):
            pass
    end = time.perf_counter()
    elapsed = end - start
    return iters / elapsed if elapsed > 0 else 0.0
