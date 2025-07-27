import numpy as np
import marble_core
from marble_core import _W1, _B1, _W2, _B2


def quantize_core_weights(num_bits: int = 8) -> None:
    """Quantize global MLP weights to the given bit precision.

    Parameters
    ----------
    num_bits:
        Number of bits to use for quantization. Must be between 1 and 16.
    """
    if not 1 <= num_bits <= 16:
        raise ValueError("num_bits must be within [1, 16]")
    qmax = 2 ** (num_bits - 1) - 1
    scale = qmax

    def _quant(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, -1.0, 1.0)
        q = np.round(clipped * scale).astype(np.int16)
        return q.astype(arr.dtype) / scale

    marble_core._W1 = _quant(_W1)
    marble_core._B1 = _quant(_B1)
    marble_core._W2 = _quant(_W2)
    marble_core._B2 = _quant(_B2)
