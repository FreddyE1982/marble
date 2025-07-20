from marble_imports import *
from marble_core import perform_message_passing, Core
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marble_neuronenblitz import Neuronenblitz

class ContrastiveLearner:
    """Self-supervised contrastive learning using InfoNCE."""

    def __init__(self, core: Core, nb: 'Neuronenblitz', temperature: float = 0.5, augment_fn=None) -> None:
        self.core = core
        self.nb = nb
        self.temperature = float(temperature)
        self.augment_fn = augment_fn if augment_fn is not None else (lambda x: x)
        self._base_update_fn = nb.weight_update_fn

    def _embed(self, inp):
        out, path = self.nb.dynamic_wander(inp)
        perform_message_passing(self.core)
        if path:
            rep = self.core.neurons[path[-1].target].representation
        else:
            rep = cp.zeros(self.core.rep_size, dtype=float)
        return rep, path

    def train(self, batch: list) -> float:
        """Perform one contrastive learning step on ``batch``."""
        views = []
        for x in batch:
            views.append(self.augment_fn(x))
            views.append(self.augment_fn(x))
        reps = []
        paths = []
        for v in views:
            rep, path = self._embed(v)
            reps.append(cp.asarray(rep))
            paths.append(path)
        reps = cp.stack(reps)
        norms = cp.linalg.norm(reps, axis=1, keepdims=True) + 1e-8
        reps = reps / norms
        sim = reps @ reps.T
        exp_sim = cp.exp(sim / self.temperature)
        n = len(batch)
        losses = []
        for i in range(n):
            pos1 = exp_sim[2*i, 2*i+1]
            denom1 = cp.sum(exp_sim[2*i]) - exp_sim[2*i, 2*i]
            losses.append(-cp.log(pos1 / denom1))
            pos2 = exp_sim[2*i+1, 2*i]
            denom2 = cp.sum(exp_sim[2*i+1]) - exp_sim[2*i+1, 2*i+1]
            losses.append(-cp.log(pos2 / denom2))
        loss = cp.mean(cp.stack(losses))
        loss_val = float(cp.asnumpy(loss))
        def safe_update(src, err, plen):
            if src is None:
                src = 0.0
            return self._base_update_fn(src, err, plen)
        prev_fn = self.nb.weight_update_fn
        self.nb.weight_update_fn = safe_update
        for p in paths:
            self.nb.apply_weight_updates_and_attention(p, loss_val)
        self.nb.weight_update_fn = prev_fn
        return loss_val
