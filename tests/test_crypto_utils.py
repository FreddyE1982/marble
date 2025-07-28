import time
from crypto_utils import constant_time_compare


def test_constant_time_compare_equal():
    assert constant_time_compare("abc", "abc")


def test_constant_time_compare_not_equal():
    assert not constant_time_compare("abc", "abd")


def test_constant_time_compare_timing():
    a = "a" * 32
    b = "b" * 32
    start = time.perf_counter_ns()
    constant_time_compare(a, a)
    equal_time = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    constant_time_compare(a, b)
    diff_time = time.perf_counter_ns() - start
    assert abs(equal_time - diff_time) < 5_000_000  # 5 ms

from crypto_profile import profile_compare

def test_profile_compare():
    diff = profile_compare(trials=1000)
    assert diff < 1000  # ns
