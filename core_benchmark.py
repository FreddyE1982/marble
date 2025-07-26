import time
from typing import Tuple

from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params


def run_benchmark(num_neurons: int = 100, iterations: int = 100) -> Tuple[int, float]:
    """Benchmark message passing speed.

    Returns a tuple ``(iterations, seconds_per_iteration)``.
    """
    params = minimal_params()
    params["representation_size"] = 4
    params["width"] = num_neurons
    params["height"] = 1
    core = Core(params)
    for n in core.neurons:
        n.representation = n.representation + 1.0

    start = time.time()
    for _ in range(iterations):
        perform_message_passing(core)
    total = time.time() - start
    return iterations, total / iterations


if __name__ == "__main__":
    iters, sec = run_benchmark()
    print(f"Ran {iters} iterations in {sec:.6f} seconds per iteration")
