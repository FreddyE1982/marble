import importlib

import torch

import predictive_coding


def test_predictive_coding_step():
    importlib.reload(predictive_coding)
    pc = predictive_coding.activate(num_layers=2, latent_dim=4, learning_rate=0.01)
    x = torch.randn(1, 4)
    err1 = pc.step(x)
    err2 = pc.step(x)
    assert err1.shape == x.shape
    assert torch.mean(err2**2) <= torch.mean(err1**2)
