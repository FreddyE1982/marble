from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class ContinuousWeightFieldLearner:
    """Continuous Weight Field Learning integrated with MARBLE."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        num_basis: int = 10,
        bandwidth: float = 1.0,
        reg_lambda: float = 0.01,
        learning_rate: float = 0.01,
    ) -> None:
        self.core = core
        self.nb = nb
        self.num_basis = int(num_basis)
        self.bandwidth = float(bandwidth)
        self.reg_lambda = float(reg_lambda)
        self.learning_rate = float(learning_rate)
        self.dim = self.core.rep_size
        rs = np.random.RandomState(self.core.params.get("random_seed", 0))
        self.centers = cp.asarray(rs.uniform(-1.0, 1.0, size=(self.num_basis, 1)))
        self.weights = cp.zeros((self.num_basis, self.dim), dtype=float)
        self.history: list[dict] = []

    def _basis(self, x: float) -> cp.ndarray:
        diff = x - self.centers
        return cp.exp(-(diff ** 2) / (2 * self.bandwidth ** 2)).reshape(-1)

    def _weight_vector(self, x: float) -> cp.ndarray:
        phi = self._basis(x).reshape(1, -1)
        return cp.dot(phi, self.weights).reshape(-1)

    def train_step(self, inp: float, target: float) -> float:
        rep_output, path = self.nb.dynamic_wander(inp)
        perform_message_passing(self.core)
        if path:
            phi_x = cp.asarray(
                self.core.neurons[path[-1].target].representation, dtype=float
            )
        else:
            phi_x = cp.zeros(self.dim, dtype=float)
        w_x = self._weight_vector(inp)
        pred = cp.dot(phi_x, w_x)
        error = float(target) - float(pred)
        basis_vals = self._basis(inp)
        for j in range(self.dim):
            grad = basis_vals * error * float(phi_x[j])
            self.weights[:, j] += self.learning_rate * grad
            self.weights[:, j] -= self.learning_rate * self.reg_lambda * self.weights[:, j]
        self.nb.apply_weight_updates_and_attention(path, error)
        loss = float(error * error)
        self.history.append({"input": float(inp), "target": float(target), "loss": loss})
        return loss

    def train(self, examples: list[tuple[float, float]], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for x, y in examples:
                self.train_step(float(x), float(y))
