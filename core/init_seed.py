from __future__ import annotations

import functools
from typing import Any, Dict

import numpy as np

import tensor_backend as tb


@functools.lru_cache(maxsize=32)
def compute_mandelbrot(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    max_iter: int = 256,
    *,
    escape_radius: float = 2.0,
    power: int = 2,
    as_numpy: bool = False,
):
    """Return a Mandelbrot set fragment using the active tensor backend."""
    xp = tb.xp()
    x = xp.linspace(xmin, xmax, width)
    y = xp.linspace(ymin, ymax, height)
    X, Y = xp.meshgrid(x, y)
    C = X + 1j * Y
    Z = xp.zeros_like(C, dtype=xp.complex64)
    mandel = xp.zeros(C.shape, dtype=xp.int32)
    esc_sq = escape_radius * escape_radius
    for i in range(int(max_iter)):
        mask = (Z.real * Z.real + Z.imag * Z.imag) <= esc_sq
        if not bool(xp.any(mask)):
            break
        Z = xp.where(mask, Z ** power + C, Z)
        mandel = xp.where(mask, i, mandel)
    if as_numpy:
        return tb.to_numpy(mandel)
    return mandel


def generate_seed(params: Dict[str, Any]) -> np.ndarray:
    """Generate initial neuron values based on Mandelbrot seeds.

    Parameters
    ----------
    params:
        Configuration dictionary containing Mandelbrot bounds and noise level.
        Expected keys include ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``width``,
        ``height`` and optional ``max_iter``, ``mandelbrot_escape_radius``,
        ``mandelbrot_power`` and ``init_noise_std``.
    """
    arr = compute_mandelbrot(
        params["xmin"],
        params["xmax"],
        params["ymin"],
        params["ymax"],
        params["width"],
        params["height"],
        params.get("max_iter", 256),
        escape_radius=params.get("mandelbrot_escape_radius", 2.0),
        power=params.get("mandelbrot_power", 2),
        as_numpy=False,
    )
    noise_std = params.get("init_noise_std", 0.0)
    if noise_std:
        noise = tb.randn(arr.shape)
        arr = arr + noise * noise_std
    return tb.to_numpy(arr)
