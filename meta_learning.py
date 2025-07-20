from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz
import copy


class MetaLearner:
    """Reptile-style meta learning across multiple tasks."""

    def __init__(self, core: Core, nb: Neuronenblitz, inner_steps: int = 1, meta_lr: float = 0.1) -> None:
        self.core = core
        self.nb = nb
        self.inner_steps = int(inner_steps)
        self.meta_lr = float(meta_lr)
        self.history: list[dict] = []

    def _get_weights(self, core: Core) -> list[float]:
        return [syn.weight for syn in core.synapses]

    def _set_weights(self, core: Core, weights: list[float]) -> None:
        for syn, w in zip(core.synapses, weights):
            syn.weight = float(w)

    def train_step(self, tasks: list[list[tuple[float, float]]]) -> float:
        if not tasks:
            raise ValueError("at least one task required")
        orig = self._get_weights(self.core)
        task_weights = []
        losses = []
        for t in tasks:
            c = copy.deepcopy(self.core)
            nb = copy.deepcopy(self.nb)
            nb.core = c
            for _ in range(self.inner_steps):
                for inp, tgt in t:
                    out, path = nb.dynamic_wander(inp)
                    err = tgt - out
                    nb.apply_weight_updates_and_attention(path, err)
                    perform_message_passing(c)
            task_weights.append(self._get_weights(c))
            val = 0.0
            for inp, tgt in t:
                pred, _ = nb.dynamic_wander(inp)
                val += float((tgt - pred) ** 2)
            losses.append(val / len(t))
        avg = np.mean(task_weights, axis=0)
        for syn, w_old, w_new in zip(self.core.synapses, orig, avg):
            syn.weight = float(w_old + self.meta_lr * (w_new - w_old))
        perform_message_passing(self.core)
        loss = float(np.mean(losses))
        self.history.append({"loss": loss})
        return loss
