import random

import torch
import torch.nn as nn

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from plugin_system import register_loss_module
from tests.test_core_functions import minimal_params


def test_neuronenblitz_accepts_loss_module():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    loss_mod = nn.MSELoss()
    nb = Neuronenblitz(core, loss_module=loss_mod)
    out, err, _ = nb.train_example(0.5, 0.2)
    expected = loss_mod(
        torch.tensor([out], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    ).item()
    assert abs(err - expected) < 1e-6


def test_neuronenblitz_accepts_loss_module_name():
    class L1Loss(nn.Module):
        def forward(self, pred, target):
            return nn.functional.l1_loss(pred, target)

    register_loss_module("l1", L1Loss)
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, loss_module="l1")
    out, err, _ = nb.train_example(0.5, 0.2)
    expected = L1Loss()(
        torch.tensor([out], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    ).item()
    assert abs(err - expected) < 1e-6
