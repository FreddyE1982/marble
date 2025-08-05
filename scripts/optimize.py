#!/usr/bin/env python3
"""Hyperparameter optimisation using Optuna.

This script performs a small Optuna study training a simple model for one
epoch on synthetic data. Results and the Optuna study are persisted to disk
so that studies can be resumed and analysed later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create training and validation loaders using ``FakeData``.

    The dataset is intentionally tiny so that optimisation runs quickly. A
    separate validation loader is returned for computing the objective value.
    """

    transform = transforms.ToTensor()
    dataset = datasets.FakeData(
        size=120,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=transform,
    )
    train_set, val_set = random_split(dataset, [100, 20])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    return train_loader, val_loader


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    """Train ``model`` for a single epoch."""

    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def _validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Return average validation loss for ``model``."""

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total += criterion(outputs, targets).item()
            count += 1
    return total / max(1, count)


def _objective(trial: optuna.Trial) -> float:
    """Optuna objective returning validation loss."""

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_int("hidden", 32, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    train_loader, val_loader = _build_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    _train_one_epoch(model, train_loader, criterion, optimizer, device)
    return _validate(model, val_loader, criterion, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation")
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--study-name", type=str, default="marble-optuna", help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_db.sqlite3",
        help="Storage URL for Optuna study (e.g. sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("best_params.yaml"),
        help="File to write best parameters as YAML",
    )
    args = parser.parse_args()

    storage = optuna.storages.RDBStorage(args.storage)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(_objective, n_trials=args.trials)

    args.output.write_text(yaml.safe_dump(study.best_trial.params))
    print(f"Best trial saved to {args.output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
