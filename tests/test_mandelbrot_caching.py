import tensor_backend as tb
from marble_core import compute_mandelbrot


def test_mandelbrot_cache_hits():
    tb.set_backend("numpy")
    compute_mandelbrot.cache_clear()
    compute_mandelbrot(-2, 1, -1.5, 1.5, 4, 4)
    before = compute_mandelbrot.cache_info().hits
    compute_mandelbrot(-2, 1, -1.5, 1.5, 4, 4)
    after = compute_mandelbrot.cache_info().hits
    assert after == before + 1
