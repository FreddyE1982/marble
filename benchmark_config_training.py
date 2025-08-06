"""Benchmark MARBLE training across multiple YAML configurations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import torch

from config_loader import create_marble_from_config

Dataset = Iterable[Tuple[float, float]]


def benchmark_training_configs(
    dataset: Dataset,
    config_paths: List[str | os.PathLike],
    *,
    snapshot: str | None = None,
    epochs: int = 1,
) -> Tuple[Dict[str, float], str]:
    """Train and evaluate MARBLE models for each configuration.

    Parameters
    ----------
    dataset:
        Iterable of ``(input, target)`` pairs used for both training and validation.
    config_paths:
        List of paths to YAML configuration files. Each file is evaluated
        independently. The best performing configuration will be renamed to
        ``best_config.yaml`` within its directory.
    snapshot:
        Optional checkpoint path. When provided, each MARBLE instance will load
        this checkpoint before training, ensuring identical initial state across
        configurations.
    epochs:
        Number of epochs to train for each configuration. Defaults to ``1``.

    Returns
    -------
    Tuple containing a mapping of configuration paths to final validation loss
    and the path of the best configuration renamed to ``best_config.yaml``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(device)
    results: Dict[str, float] = {}

    for cfg in config_paths:
        cfg_path = Path(cfg)
        marble = create_marble_from_config(str(cfg_path))
        brain = marble.get_brain()
        if snapshot is not None:
            brain.load_checkpoint(snapshot)
        brain.train(dataset, epochs=epochs, validation_examples=dataset)
        loss = float(brain.validate(dataset))
        results[str(cfg_path)] = loss
        print(f"{cfg_path.name}: {loss:.6f}")
        del marble
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg_names = list(results.keys())
    for i, a in enumerate(cfg_names):
        for b in cfg_names[i + 1 :]:
            diff = results[a] - results[b]
            print(f"{Path(a).name} vs {Path(b).name}: {diff:+.6f}")

    best_cfg = min(results, key=results.get)
    best_path = Path(best_cfg)
    target = best_path.with_name("best_config.yaml")
    if target.exists():
        target.unlink()
    os.rename(best_path, target)
    print(f"Best configuration: {best_path.name} -> {target.name}")
    return results, str(target)


__all__ = ["benchmark_training_configs"]
