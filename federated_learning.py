from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class FederatedAveragingTrainer:
    """Federated averaging across multiple Neuronenblitz clients."""

    def __init__(self, clients: list[tuple[Core, Neuronenblitz]]) -> None:
        if not clients:
            raise ValueError("at least one client required")
        self.clients = clients

    def _get_weights(self, core: Core) -> list[float]:
        return [syn.weight for syn in core.synapses]

    def _set_weights(self, core: Core, weights: list[float]) -> None:
        for syn, w in zip(core.synapses, weights):
            syn.weight = float(w)

    def _average_weights(self) -> list[float]:
        weight_lists = [self._get_weights(c) for c, _ in self.clients]
        if not weight_lists:
            return []
        min_len = min(len(w) for w in weight_lists)
        trimmed = [w[:min_len] for w in weight_lists]
        return list(np.mean(trimmed, axis=0))

    def aggregate(self) -> None:
        """Average synapse weights across clients."""
        avg = self._average_weights()
        for c, _ in self.clients:
            self._set_weights(c, avg)

    def train_round(self, datasets: list[list[tuple[float, float]]], epochs: int = 1) -> None:
        """Train each client locally then aggregate weights."""
        if len(datasets) != len(self.clients):
            raise ValueError("datasets length must match clients")
        for (core, nb), data in zip(self.clients, datasets):
            nb.train(data, epochs=epochs)
            perform_message_passing(core)
        self.aggregate()
