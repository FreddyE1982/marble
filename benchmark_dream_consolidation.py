"""Benchmark impact of dream consolidation on learning performance.

This script trains a small Neuronenblitz learner on a synthetic task
and compares learning with and without dream-based consolidation.
It reports average error and runtime for both settings.
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

from dream_reinforcement_learning import DreamReinforcementLearner
from marble_core import Core
from marble_imports import cp
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def _generate_dataset(num_examples: int) -> List[Tuple[float, float]]:
    """Generate a simple linear dataset ``y = 2x`` for training."""
    xs = cp.linspace(0, 1, num_examples)
    ys = 2 * xs
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def _run_cycle(dream_cycles: int, episodes: int) -> Dict[str, float]:
    """Run training for a given number of dream cycles and collect metrics."""
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = DreamReinforcementLearner(core, nb, dream_cycles=dream_cycles)
    dataset = _generate_dataset(episodes)

    start = time.time()
    learner.train(dataset, repeat=1)
    duration = time.time() - start

    errors = [h["error"] for h in learner.history]
    avg_error = float(statistics.fmean(errors)) if errors else float("nan")
    final_error = float(errors[-1]) if errors else float("nan")

    return {
        "dream_cycles": dream_cycles,
        "avg_error": avg_error,
        "final_error": final_error,
        "duration": duration,
    }


def run_benchmark(episodes: int = 50) -> Dict[str, Dict[str, float]]:
    """Compare learning with and without dream consolidation."""
    with_dream = _run_cycle(dream_cycles=1, episodes=episodes)
    without_dream = _run_cycle(dream_cycles=0, episodes=episodes)
    return {"with_dream": with_dream, "without_dream": without_dream}


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    results = run_benchmark()
    print("Dream consolidation benchmark results:")
    for label, metrics in results.items():
        print(
            f"{label} -> avg_error: {metrics['avg_error']:.6f}, "
            f"final_error: {metrics['final_error']:.6f}, "
            f"duration: {metrics['duration']:.4f}s"
        )
