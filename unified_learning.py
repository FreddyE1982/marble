import json
import random
from typing import Dict, List, Callable

import torch

from marble_core import Core, perform_message_passing
from marble_neuronenblitz import Neuronenblitz


class UnifiedLearner:
    """Meta controller coordinating multiple MARBLE paradigms."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        learners: Dict[str, object],
        gating_hidden: int = 8,
        log_path: str | None = None,
    ) -> None:
        self.core = core
        self.nb = nb
        self.learners = learners
        self.log_path = log_path
        in_dim = 4
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_dim, gating_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(gating_hidden, len(learners)),
        )
        self.loss_history: Dict[str, List[float]] = {n: [] for n in learners}
        self.history: List[dict] = []

    def _context_vector(self) -> torch.Tensor:
        avg_loss = (
            sum(v[-1] for v in self.loss_history.values() if v)
            / max(1, sum(1 for v in self.loss_history.values() if v))
        )
        ctx = [
            float(self.core.get_usage_by_tier("vram")),
            float(self.nb.plasticity_threshold),
            float(len(self.core.neurons)),
            avg_loss,
        ]
        return torch.tensor(ctx, dtype=torch.float32)

    def _select_weights(self, ctx: torch.Tensor) -> torch.Tensor:
        logits = self.gate(ctx)
        return torch.softmax(logits, dim=0)

    def _log(self, ctx: torch.Tensor, weights: torch.Tensor) -> None:
        entry = {
            "context": ctx.detach().cpu().tolist(),
            "weights": {n: float(w) for n, w in zip(self.learners, weights)},
        }
        self.history.append(entry)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def train_step(self, sample: tuple[float, float]) -> None:
        inp, target = sample
        ctx = self._context_vector()
        weights = self._select_weights(ctx)
        self._log(ctx, weights)
        for weight, (name, learner) in zip(weights, self.learners.items()):
            prev = getattr(self.nb, "plasticity_modulation", 1.0)
            self.nb.plasticity_modulation = float(weight)
            if hasattr(learner, "train_step"):
                try:
                    loss = learner.train_step(inp, target)
                except TypeError:
                    loss = learner.train_step(inp)
            else:
                continue
            self.loss_history[name].append(float(loss) if loss is not None else 0.0)
            self.nb.plasticity_modulation = prev
        perform_message_passing(self.core)

    def explain(self, index: int, with_gradients: bool = False) -> dict:
        """Return logged context and weights for a step.

        When ``with_gradients`` is ``True`` this also computes the gradient of
        each learner's weight with respect to the context features. The
        resulting dictionary then contains an additional ``"gradients"`` entry
        mapping learner names to lists of contributions for every context
        element.
        """

        if not (0 <= index < len(self.history)):
            return {}

        entry = dict(self.history[index])
        if not with_gradients:
            return entry

        ctx = torch.tensor(entry["context"], dtype=torch.float32, requires_grad=True)
        logits = self.gate(ctx)
        weights = torch.softmax(logits, dim=0)
        grads: Dict[str, List[float]] = {}
        for i, name in enumerate(self.learners):
            self.gate.zero_grad()
            if ctx.grad is not None:
                ctx.grad.zero_()
            weights[i].backward(retain_graph=True)
            grads[name] = ctx.grad.detach().cpu().tolist()
        entry["gradients"] = grads
        return entry
