import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_neuronenblitz as nb_mod
from benchmark_super_evolution import run_benchmark


def test_super_evolution_benchmark_structure():
    nb_mod.print = lambda *a, **k: None
    results = run_benchmark(num_runs=2, seed=0)
    assert len(results) == 2
    for run in results:
        assert len(run) == 20
        for entry in run:
            assert "loss" in entry and "changes" in entry
            assert isinstance(entry["loss"], float)
            assert isinstance(entry["changes"], list)
        assert any(entry["changes"] for entry in run)
