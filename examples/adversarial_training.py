"""Example adversarial training loop using Neuronenblitz."""

from __future__ import annotations

import torch
from torch import nn
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from adversarial_utils import fgsm_attack


def main() -> None:
    core = Core({})
    nb = Neuronenblitz(core)
    model = nn.Linear(1, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    inputs = torch.randn(10, 1)
    labels = torch.randint(0, 2, (10,))

    for _ in range(5):
        optim.zero_grad()
        adv_inputs = fgsm_attack(model, inputs, labels)
        out = model(adv_inputs)
        loss = nn.CrossEntropyLoss()(out, labels)
        loss.backward()
        optim.step()
    print("Finished adversarial training")


if __name__ == "__main__":
    main()
