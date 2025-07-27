import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import marble_core
from model_quantization import quantize_core_weights


def test_quantize_core_weights_reduces_precision():
    orig_unique = len(np.unique(marble_core._W1))
    quantize_core_weights(4)
    new_unique = len(np.unique(marble_core._W1))
    assert new_unique <= orig_unique
