import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core_benchmark import run_benchmark


def test_message_passing_speed():
    _, sec = run_benchmark(num_neurons=50, iterations=20)
    assert sec < 0.05

