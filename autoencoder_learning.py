from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class AutoencoderLearner:
    """Denoising autoencoder training using Neuronenblitz and MARBLE Core."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        noise_std: float = 0.1,
        noise_decay: float = 0.99,
    ) -> None:
        self.core = core
        self.nb = nb
        self.noise_std = float(noise_std)
        self.noise_decay = float(noise_decay)
        self.history: list[dict] = []

    def _noisy_input(self, value: float) -> float:
        return float(value + np.random.normal(0.0, self.noise_std))

    def train_step(self, value: float) -> float:
        noisy = self._noisy_input(value)
        out, path = self.nb.dynamic_wander(noisy)
        error = value - out
        self.nb.apply_weight_updates_and_attention(path, error)
        perform_message_passing(self.core)
        loss = float(error * error)
        self.history.append({"input": value, "reconstructed": out, "loss": loss})
        return loss

    def train(self, values: list[float], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for v in values:
                self.train_step(float(v))
            self.noise_std *= self.noise_decay
