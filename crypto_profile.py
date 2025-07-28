import time
from crypto_utils import constant_time_compare


def profile_compare(trials: int = 10000) -> float:
    """Return average time difference between equal and non-equal comparisons."""
    a = b"a" * 32
    c = b"b" * 32
    start = time.perf_counter_ns()
    for _ in range(trials):
        constant_time_compare(a, a)
    equal_time = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    for _ in range(trials):
        constant_time_compare(a, c)
    diff_time = time.perf_counter_ns() - start
    return abs(equal_time - diff_time) / trials


if __name__ == "__main__":
    diff = profile_compare()
    print(f"Average time difference per call: {diff:.2f} ns")
