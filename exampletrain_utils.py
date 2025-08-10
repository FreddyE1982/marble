"""Utility helpers for ``exampletrain`` tests.

This module provides lightweight fixtures that allow the heavy
``exampletrain.py`` script to be tested without downloading large
datasets or initialising the full Stable Diffusion pipeline.  The
utilities here are intentionally self‑contained and operate entirely on
CPU or GPU depending on availability so that parity between devices can
be verified easily.

Currently the module exposes :func:`create_synthetic_dataset` which
produces a deterministic sequence of input/target pairs.  Each value is
generated using the supplied random seed so tests can rely on
reproducible behaviour.  When a CUDA device is available the tensors are
allocated on that device; otherwise they default to the CPU.  Callers
can also explicitly select the device.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def create_synthetic_dataset(
    num_samples: int = 10,
    device: Optional[str] = None,
    seed: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a deterministic synthetic dataset.

    Parameters
    ----------
    num_samples:
        Number of ``(input, target)`` pairs to produce.
    device:
        The device on which tensors should be allocated.  If ``None`` the
        function will automatically select ``"cuda"`` when available or
        fallback to ``"cpu"`` otherwise.
    seed:
        Random seed controlling the generated values.

    Returns
    -------
    list of tuple of ``torch.Tensor``
        Each tuple contains an ``input`` tensor and a corresponding
        ``target`` tensor.  The relationship ``target = 2 * input + 1`` is
        intentionally simple yet non‑trivial, enabling tests to verify
        training and inference logic without relying on external
        resources.
    """

    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=actual_device)
    gen.manual_seed(seed)

    inputs = torch.rand(num_samples, generator=gen, device=actual_device)
    targets = 2 * inputs + 1

    return list(zip(inputs, targets))


__all__ = ["create_synthetic_dataset"]
