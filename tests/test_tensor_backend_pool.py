import numpy as np
import pytest

import tensor_backend as tb
from memory_pool import ArrayMemoryPool


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_matmul_reuses_pool(backend):
    try:
        tb.set_backend(backend)
    except ImportError:
        pytest.skip("JAX not installed")
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    pool = ArrayMemoryPool((2, 2), dtype=np.float32)
    result1 = tb.matmul(a, b, out_pool=pool)
    pool.release(result1)
    result2 = tb.matmul(a, b, out_pool=pool)
    assert result1 is result2
    pool.release(result2)


def test_activation_ops_pool():
    tb.set_backend("numpy")
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    pool = ArrayMemoryPool((3,), dtype=np.float32)
    sig = tb.sigmoid(x, out_pool=pool)
    pool.release(sig)
    rel = tb.relu(x, out_pool=pool)
    assert np.all(rel >= 0)
    pool.release(rel)
