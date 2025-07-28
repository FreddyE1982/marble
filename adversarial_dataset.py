"""Dataset wrappers for generating adversarial examples on the fly."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FGSMDataset(Dataset):
    """Wrap another dataset to generate FGSM adversarial samples.

    Parameters
    ----------
    dataset:
        The underlying dataset providing ``(input, target)`` pairs.
    model:
        A PyTorch model used to compute input gradients.
    epsilon:
        Magnitude of the adversarial perturbation.
    device:
        Device to run computations on. Defaults to CUDA when available.
    loss_fn:
        Loss function to use for computing gradients. Defaults to
        ``torch.nn.MSELoss``.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        *,
        epsilon: float = 0.01,
        device: str | torch.device | None = None,
        loss_fn: torch.nn.Module | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.epsilon = float(epsilon)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.model.to(self.device)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.dataset)

    def _to_tensor(self, value: torch.Tensor | float | int | list | tuple) -> torch.Tensor:
        t = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        x_t = self._to_tensor(x).clone().detach().requires_grad_(True)
        y_t = self._to_tensor(y)

        output = self.model(x_t.unsqueeze(0))
        loss = self.loss_fn(output.squeeze(), y_t.squeeze())
        loss.backward()
        grad = x_t.grad.detach()
        adv_x = (x_t + self.epsilon * grad.sign()).detach()

        x_out = adv_x.squeeze().cpu().numpy()
        if x_out.shape == ():
            x_out = float(x_out)
        return x_out, y
