"""Utilities for deterministic k-fold cross-validation.

This module provides helpers to split datasets into deterministic
train/validation folds and to execute cross-validation loops that work
seamlessly on CPU or GPU. The splits are created using a fixed random
seed to ensure repeatability across runs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Callable, Iterable, List, Tuple


import torch
from torch.utils.data import Subset


def k_fold_split(
    dataset: Sequence, k: int, seed: int = 0
) -> List[Tuple[Subset, Subset]]:
    """Return ``k`` deterministic train/validation splits of ``dataset``."""
    if k <= 1:
        raise ValueError("k must be greater than 1")
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    folds: List[Tuple[Subset, Subset]] = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current : current + fold_size]
        train_idx = indices[:current] + indices[current + fold_size :]
        folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx)))
        current += fold_size
    return folds


def cross_validate(
    train_fn: Callable[[Iterable, torch.device], Any],
    metric_fn: Callable[[Any, Iterable, torch.device], float],
    dataset: Sequence,
    *,
    folds: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> List[float]:
    """Run k-fold cross-validation on ``dataset``.

    ``folds`` and ``seed`` default to the ``cross_validation`` section of
    :mod:`config.yaml` when unspecified.
    """

    if folds is None or seed is None:
        from config_loader import load_config

        cfg = load_config()
        cv_cfg = cfg.get("cross_validation", {})
        if folds is None:
            folds = int(cv_cfg.get("folds", 5))
        if seed is None:
            seed = int(cv_cfg.get("seed", 0))

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: List[float] = []
    for train_set, val_set in k_fold_split(dataset, folds, seed):
        model = train_fn(_move_batches(train_set, device), device)
        score = metric_fn(model, _move_batches(val_set, device), device)
        scores.append(float(score))
    return scores


def _move_batches(dataset: Iterable, device: torch.device) -> List[Any]:
    """Return dataset items with tensors moved to ``device``."""
    moved: List[Any] = []
    for item in dataset:
        moved.append(_move_to_device(item, device))
    return moved


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj
