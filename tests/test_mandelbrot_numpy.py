import numpy as np
from marble_core import compute_mandelbrot


def test_compute_mandelbrot_as_numpy():
    arr = compute_mandelbrot(-2, 1, -1.5, 1.5, 3, 3, as_numpy=True)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 3)
