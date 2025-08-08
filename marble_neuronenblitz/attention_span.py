import torch

class DynamicSpanModule:
    """Adaptive attention span selection for Neuronenblitz.

    The module scores attention weights and selects a variable span per batch
    based on a cumulative probability threshold. It supports batched inputs and
    automatically utilises CUDA when available, falling back to CPU otherwise.
    """

    def __init__(self, max_span: int | None = None, threshold: float = 0.9, device: torch.device | None = None) -> None:
        self.max_span = max_span
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score_spans(self, scores: torch.Tensor) -> torch.Tensor:
        """Return cumulative attention distribution for ``scores``.

        Parameters
        ----------
        scores:
            Tensor of shape ``(batch, seq_len)`` containing raw attention
            scores. Higher values denote greater importance.
        """

        norm = torch.softmax(scores, dim=-1)
        return torch.cumsum(norm, dim=-1)

    def select_spans(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute a boolean mask selecting the adaptive span.

        The mask has the same shape as ``scores`` and marks elements that lie
        within the dynamic span for each batch item.
        """

        cum_scores = self.score_spans(scores)
        mask = cum_scores <= self.threshold
        mask[..., 0] = True  # always keep at least one element
        if self.max_span is not None:
            idx = torch.arange(scores.size(-1), device=scores.device)
            mask = mask & (idx < self.max_span)
        return mask

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        """Return selection mask for ``scores`` on the configured device."""

        scores = scores.to(self.device)
        mask = self.select_spans(scores)
        return mask.detach().cpu()
