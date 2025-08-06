"""Utilities for attention mechanisms including causal masking and gating.

Provides functions to generate causal masks that block access to future
positions, gating layers that modulate attention scores using sine or chaotic
sequences, benchmarking helpers and plotting utilities for visualisation in
Streamlit.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import plotly.graph_objs as go
import torch


def generate_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return an upper triangular mask where ``True`` denotes masked positions.

    Parameters
    ----------
    seq_len:
        Length of the sequence.
    device:
        Torch device on which to allocate the mask.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(seq_len, seq_len)`` with ``mask[i, j]`` set to
        ``True`` whenever ``j > i``.
    """

    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )


class GatingLayer(torch.nn.Module):
    """Deterministic gating modulation for attention scores.

    The layer produces a 1D gating vector that is multiplied element-wise with
    attention logits before softmax. Two modes are available:

    ``"sine"``
        Produces a sinusoidal waveform with values in ``[-1, 1]``. The
        ``frequency`` parameter controls the number of oscillations across the
        sequence length.
    ``"chaos"``
        Generates a chaotic sequence using the logistic map ``x_{n+1} = r*x_n*(1-x_n)``
        with ``r`` specified by ``chaos``. Outputs lie in ``(0, 1)``.
    """

    def __init__(
        self,
        mode: Literal["sine", "chaos"] = "sine",
        *,
        frequency: float = 1.0,
        chaos: float = 3.7,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.frequency = frequency
        self.chaos = chaos

    def forward(self, length: int, device: torch.device | None = None) -> torch.Tensor:
        t = torch.arange(length, device=device, dtype=torch.float32)
        if self.mode == "sine":
            # Scale index so ``frequency`` represents complete waves over the sequence.
            return torch.sin(2 * torch.pi * self.frequency * t / max(length, 1))
        if self.mode == "chaos":
            out = torch.zeros(length, device=device, dtype=torch.float32)
            x = torch.tensor(0.5, device=device, dtype=torch.float32)
            for i in range(length):
                x = self.chaos * x * (1.0 - x)
                out[i] = x
            return out
        return torch.ones(length, device=device, dtype=torch.float32)


def benchmark_mask_overhead(
    seq_len: int, device: str = "cpu", repeats: int = 10
) -> float:
    """Benchmark average time to generate a causal mask on given device."""

    dev = torch.device(device)
    start = time.perf_counter()
    for _ in range(repeats):
        generate_causal_mask(seq_len, device=dev)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / repeats


def mask_figure(mask: torch.Tensor | np.ndarray) -> go.Figure:
    """Create a heatmap figure for a causal mask."""

    if isinstance(mask, torch.Tensor):
        mask = mask.to(torch.float32).cpu().numpy()
    fig = go.Figure(data=go.Heatmap(z=mask))
    fig.update_layout(title="Causal Mask", xaxis_title="Key", yaxis_title="Query")
    return fig


def gate_figure(gate: torch.Tensor | np.ndarray) -> go.Figure:
    """Create a line plot for gating values."""

    if isinstance(gate, torch.Tensor):
        gate = gate.cpu().numpy()
    fig = go.Figure(data=go.Scatter(y=gate))
    fig.update_layout(title="Gating Signal", xaxis_title="Position", yaxis_title="Gate")
    return fig

