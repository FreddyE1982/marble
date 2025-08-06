import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensor_backend as tb
from marble_core import Core, benchmark_message_passing
from tests.test_core_functions import minimal_params


def test_benchmark_message_passing_returns_tuple():
    tb.set_backend("numpy")
    params = minimal_params()
    core = Core(params)
    iters, sec = benchmark_message_passing(core, iterations=2, warmup=1)
    assert iters == 2
    assert isinstance(sec, float) and sec >= 0.0
