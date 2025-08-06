import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

import benchmark_dream_consolidation as bdc


def test_benchmark_dream_consolidation_runs():
    results = bdc.run_benchmark(episodes=10)
    assert "with_dream" in results and "without_dream" in results
    wd = results["with_dream"]
    wo = results["without_dream"]
    assert wd["avg_error"] == pytest.approx(wd["avg_error"])
    assert wo["avg_error"] == pytest.approx(wo["avg_error"])
    assert wd["duration"] >= 0.0 and wo["duration"] >= 0.0
