"""Utility functions for hyperparameter search."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Mapping


def grid_search(
    param_grid: Mapping[str, Iterable[Any]],
    train_func: Callable[[dict[str, Any]], float],
) -> list[tuple[dict[str, Any], float]]:
    """Explore all combinations in ``param_grid`` using ``train_func``.

    ``param_grid`` maps parameter names to iterables of possible values. For each
    combination ``train_func`` is called with a dictionary of parameters and
    should return a numeric score (lower is better).
    """
    keys = list(param_grid)
    values = [list(v) for v in param_grid.values()]
    results: list[tuple[dict[str, Any], float]] = []
    for combo in product(*values):
        params = {k: v for k, v in zip(keys, combo)}
        score = float(train_func(params))
        results.append((params, score))
    results.sort(key=lambda x: x[1])
    return results


def random_search(
    param_options: Mapping[str, Iterable[Any]],
    train_func: Callable[[dict[str, Any]], float],
    num_samples: int,
) -> list[tuple[dict[str, Any], float]]:
    """Randomly sample ``num_samples`` parameter sets from ``param_options``."""
    import random

    keys = list(param_options)
    options = [list(v) for v in param_options.values()]
    results: list[tuple[dict[str, Any], float]] = []
    for _ in range(num_samples):
        params = {k: random.choice(v) for k, v in zip(keys, options)}
        score = float(train_func(params))
        results.append((params, score))
    results.sort(key=lambda x: x[1])
    return results
