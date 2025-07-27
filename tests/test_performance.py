import time
import numpy as np
import marble_core


def test_simple_mlp_performance():
    x = np.random.randn(1000, marble_core._REP_SIZE)
    start = time.time()
    for _ in range(50):
        marble_core._simple_mlp(x)
    duration = time.time() - start
    assert duration < 2.0
