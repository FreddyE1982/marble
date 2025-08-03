import numpy as np
import pytest

import tensor_backend as tb


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_ops_equivalence(backend):
    try:
        tb.set_backend(backend)
    except ImportError:
        pytest.skip("JAX not installed")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    mm = tb.matmul(a, b)
    assert np.allclose(mm, a)
    x = np.array([-1.0, 0.0, 1.0])
    sig = tb.sigmoid(x)
    rel = tb.relu(x)
    assert np.all(sig >= 0) and np.all(sig <= 1)
    assert np.all(rel >= 0)
