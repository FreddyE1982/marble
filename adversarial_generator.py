"""Helpers for generating adversarial examples."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


def fgsm_generate(
    model: torch.nn.Module,
    inputs: Iterable[torch.Tensor],
    targets: Iterable[torch.Tensor],
    *,
    epsilon: float = 0.1,
    loss_fn: torch.nn.Module | None = None,
    device: str | torch.device | None = None,
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    """Return a list of FGSM adversarial examples.

    Parameters
    ----------
    model:
        PyTorch model used to compute gradients.
    inputs:
        Iterable of input tensors.
    targets:
        Iterable of target tensors.
    epsilon:
        Perturbation magnitude.
    loss_fn:
        Loss used to compute gradients. Defaults to ``nn.MSELoss``.
    device:
        Device for computation. Defaults to CUDA if available.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = loss_fn or torch.nn.MSELoss()
    adv_samples: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in zip(inputs, targets):
        x = x.to(device).detach().clone().requires_grad_(True)
        y = y.to(device)
        output = model(x.unsqueeze(0))
        loss = loss_fn(output.squeeze(), y.squeeze())
        loss.backward()
        adv_x = (x + epsilon * x.grad.sign()).detach()
        adv_samples.append((adv_x.cpu(), y.cpu()))
    return adv_samples
