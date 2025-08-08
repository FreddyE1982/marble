"""Utilities for refreshing Neuronenblitz models when datasets change.

This module exposes high level routines for performing a full retrain or
an incremental update.  Both helpers are device-aware and automatically use
CUDA when available but gracefully fall back to CPU execution.
"""
from __future__ import annotations

from typing import Iterable, Callable

import torch
from torch.utils.data import DataLoader


def _detect_device(device: str | torch.device | None = None) -> torch.device:
    """Return a valid :class:`torch.device`.

    The helper checks ``torch.cuda.is_available`` and selects the best device.
    Users may override the choice by passing ``device`` explicitly.
    """

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_retrain(
    model: torch.nn.Module,
    dataset: Iterable,
    epochs: int = 1,
    batch_size: int = 32,
    optimizer_fn: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Retrain ``model`` from scratch on ``dataset``.

    The routine resets all model parameters using ``reset_parameters`` when
    available, moves the model to the selected device and performs a basic
    supervised training loop.  ``optimizer_fn`` and ``loss_fn`` can be
    customised; they default to :class:`torch.optim.Adam` and
    :class:`torch.nn.MSELoss` respectively.
    """

    dev = _detect_device(device)
    model.to(dev)
    model.train()

    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if optimizer_fn is None:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = optimizer_fn(model.parameters())
    if loss_fn is None:
        criterion = torch.nn.MSELoss()
    else:
        criterion = loss_fn

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model


def incremental_update(
    model: torch.nn.Module,
    new_data: Iterable,
    epochs: int = 1,
    batch_size: int = 32,
    optimizer_fn: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer] | None = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Continue training ``model`` on ``new_data``.

    Unlike :func:`full_retrain`, this function preserves the existing
    parameters and trains only on the provided dataset.  The model is moved to
    the specified device and a simple training loop is executed.
    """

    dev = _detect_device(device)
    model.to(dev)
    model.train()

    if optimizer_fn is None:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        optimizer = optimizer_fn(model.parameters())
    if loss_fn is None:
        criterion = torch.nn.MSELoss()
    else:
        criterion = loss_fn

    loader = DataLoader(new_data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

