import numpy as np
from sparse_utils import to_sparse, to_dense, benchmark_memory_savings


def test_sparse_dense_roundtrip():
    mat = np.zeros((5, 5), dtype=np.float32)
    mat[0, 1] = 1.0
    sp = to_sparse(mat)
    dense = to_dense(sp)
    assert np.array_equal(mat, dense)


def test_memory_savings_positive():
    mat = np.zeros((100, 100), dtype=np.float32)
    mat[0, 0] = 1.0
    stats = benchmark_memory_savings(mat)
    assert stats["savings"] > 0
