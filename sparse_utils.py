import numpy as np
from scipy import sparse

__all__ = ["to_sparse", "to_dense", "memory_bytes", "benchmark_memory_savings"]


def to_sparse(array: np.ndarray, fmt: str = "csr") -> sparse.spmatrix:
    """Convert a dense NumPy array to a SciPy sparse matrix.

    Parameters
    ----------
    array:
        Dense array to convert.
    fmt:
        Sparse format to use. Supported values are ``"csr"``, ``"csc"`` and
        ``"coo"``. Defaults to ``"csr"``.
    """
    if fmt == "csr":
        return sparse.csr_matrix(array)
    if fmt == "csc":
        return sparse.csc_matrix(array)
    if fmt == "coo":
        return sparse.coo_matrix(array)
    raise ValueError(f"Unsupported sparse format: {fmt}")


def to_dense(matrix: sparse.spmatrix) -> np.ndarray:
    """Convert a SciPy sparse matrix back to a dense NumPy array."""
    return matrix.toarray()


def memory_bytes(matrix: sparse.spmatrix) -> int:
    """Return the number of bytes consumed by ``matrix``."""
    data_bytes = matrix.data.nbytes
    ind_bytes = matrix.indices.nbytes if hasattr(matrix, "indices") else 0
    indptr_bytes = matrix.indptr.nbytes if hasattr(matrix, "indptr") else 0
    return data_bytes + ind_bytes + indptr_bytes


def benchmark_memory_savings(array: np.ndarray, fmt: str = "csr") -> dict:
    """Benchmark memory usage of dense versus sparse representations.

    Returns a dictionary with ``dense`` and ``sparse`` byte counts along with
    the absolute ``savings`` when using the sparse representation.
    """
    sp = to_sparse(array, fmt=fmt)
    dense_bytes = array.nbytes
    sparse_bytes = memory_bytes(sp)
    return {"dense": dense_bytes, "sparse": sparse_bytes, "savings": dense_bytes - sparse_bytes}
