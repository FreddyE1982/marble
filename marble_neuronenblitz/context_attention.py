"""Context-aware attention layer for Neuronenblitz."""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_utils import GatingLayer, generate_causal_mask


class ContextAwareAttention(nn.Module):
    """Scaled dot-product attention augmented with a context vector."""

    def __init__(
        self,
        dim: int,
        *,
        causal: bool = False,
        gating: dict | None = None,
    ) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.causal = causal
        self.context = nn.Parameter(torch.zeros(dim))
        self.to_keys = nn.Linear(dim, dim)
        self.to_queries = nn.Linear(dim, dim)
        self.to_values = nn.Linear(dim, dim)
        if gating and gating.get("enabled", False):
            mode = gating.get("mode", "sine")
            freq = float(gating.get("frequency", 1.0))
            chaos = float(gating.get("chaos", 3.7))
            self.gating = GatingLayer(mode, frequency=freq, chaos=chaos)
        else:
            self.gating: GatingLayer | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return attention-weighted values.

        The computation runs on GPU when available. When ``return_weights`` is
        ``True`` a tuple ``(output, weights)`` is returned where ``weights``
        contains the attention probabilities.
        """

        device = query.device
        ctx = self.context.to(device)

        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            squeeze = True

        q = self.to_queries(query + ctx)
        k = self.to_keys(key + ctx)
        v = self.to_values(value)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = generate_causal_mask(scores.size(-1), device=device)
            scores = scores.masked_fill(mask, float("-inf"))
        if self.gating is not None:
            gate = self.gating(scores.size(-1), device=device)
            scores = scores * gate

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        if squeeze:
            out = out.squeeze(1)
            weights = weights.squeeze(1)
        return (out, weights) if return_weights else out
