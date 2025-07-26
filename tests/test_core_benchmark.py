import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_benchmark import run_benchmark


def test_core_benchmark_runs():
    iters, sec = run_benchmark(num_neurons=10, iterations=5)
    assert iters == 5
    assert sec > 0
