from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class SemiSupervisedLearner:
    """Consistency regularization using labeled and unlabeled data."""

    def __init__(self, core: Core, nb: Neuronenblitz, unlabeled_weight: float = 0.5) -> None:
        self.core = core
        self.nb = nb
        self.unlabeled_weight = float(unlabeled_weight)
        self.history: list[dict] = []

    def train_step(self, labeled_pair: tuple[float, float], unlabeled_input: float) -> float:
        labeled_inp, target = labeled_pair
        out, path = self.nb.dynamic_wander(labeled_inp)
        error = target - out
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)

        pred1, path1 = self.nb.dynamic_wander(unlabeled_input)
        pred2, path2 = self.nb.dynamic_wander(unlabeled_input)
        cons_error = pred1 - pred2
        self.nb.apply_weight_updates_and_attention(path1, self.unlabeled_weight * cons_error)
        self.nb.apply_weight_updates_and_attention(path2, -self.unlabeled_weight * cons_error)
        perform_message_passing(self.core)

        loss = error * error + self.unlabeled_weight * cons_error * cons_error
        self.history.append({"sup_loss": float(error * error), "consistency": float(cons_error * cons_error)})
        return float(loss)

    def train(
        self,
        labeled_pairs: list[tuple[float, float]],
        unlabeled_inputs: list[float],
        epochs: int = 1,
    ) -> None:
        for _ in range(int(epochs)):
            for lp, ui in zip(labeled_pairs, unlabeled_inputs):
                self.train_step(lp, ui)
