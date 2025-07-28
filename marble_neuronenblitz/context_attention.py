"""Context-aware attention layer for Neuronenblitz."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextAwareAttention(nn.Module):
    """Scaled dot-product attention augmented with a context vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.context = nn.Parameter(torch.zeros(dim))
        self.to_keys = nn.Linear(dim, dim)
        self.to_queries = nn.Linear(dim, dim)
        self.to_values = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Return attention-weighted values.

        The computation runs on GPU when available.
        """
        device = query.device
        ctx = self.context.to(device)
        q = self.to_queries(query + ctx)
        k = self.to_keys(key + ctx)
        v = self.to_values(value)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)
