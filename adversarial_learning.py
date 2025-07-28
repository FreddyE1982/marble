from marble_imports import *
from marble_core import perform_message_passing, Core
from marble_neuronenblitz import Neuronenblitz


class AdversarialLearner:
    """Simple GAN-style training using Neuronenblitz networks."""

    def __init__(self, core: Core, generator: Neuronenblitz, discriminator: Neuronenblitz, noise_dim: int = 1) -> None:
        self.core = core
        self.generator = generator
        self.discriminator = discriminator
        self.noise_dim = int(noise_dim)
        self.history: list[dict] = []

    def _sample_noise(self) -> float:
        return float(np.random.randn(self.noise_dim).mean())

    def train_step(self, real_value: float) -> float:
        noise = self._sample_noise()
        fake_out, gen_path = self.generator.dynamic_wander(noise)
        gen_sources = [self.core.neurons[s.source].value for s in gen_path]

        real_pred, real_path = self.discriminator.dynamic_wander(real_value)
        self.discriminator.apply_weight_updates_and_attention(real_path, 1.0 - real_pred)
        fake_pred, fake_path = self.discriminator.dynamic_wander(fake_out)
        self.discriminator.apply_weight_updates_and_attention(fake_path, 0.0 - fake_pred)
        perform_message_passing(self.core)

        fake_pred2, _ = self.discriminator.dynamic_wander(fake_out)
        gen_error = 1.0 - fake_pred2
        # restore generator source values before applying updates
        for syn, val in zip(gen_path, gen_sources):
            self.core.neurons[syn.source].value = val
        self.generator.apply_weight_updates_and_attention(gen_path, gen_error)
        perform_message_passing(self.core)
        self.history.append({
            "real": real_value,
            "fake": fake_out,
            "real_pred": real_pred,
            "fake_pred": fake_pred,
            "gen_error": gen_error,
        })
        return float(gen_error)

    def train(self, real_values: list[float], epochs: int = 1) -> None:
        for _ in range(int(epochs)):
            for val in real_values:
                self.train_step(float(val))


def train_with_adversarial_examples(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    epsilon: float = 0.1,
    epochs: int = 1,
    device: str | torch.device | None = None,
) -> None:
    """Simple FGSM adversarial training loop for a PyTorch model."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model.to(device)
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device).detach().clone().requires_grad_(True)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            adv_x = (x + epsilon * x.grad.sign()).detach()
            adv_out = model(adv_x)
            adv_loss = loss_fn(adv_out, y)
            adv_loss.backward()
            opt.step()
