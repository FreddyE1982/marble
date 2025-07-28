"""Utilities for generating adversarial examples."""

from __future__ import annotations

import torch
from torch import nn


def fgsm_attack(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """Generate FGSM adversarial examples for ``model``."""
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = (inputs + perturbation).clamp(0, 1)
    return adv_inputs.detach()
