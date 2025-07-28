"""Predictive Coding plugin implementing a simple hierarchical predictor."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

import global_workspace


class PredictiveCodingLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.state = nn.Parameter(torch.zeros(latent_dim))
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.encoder = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.decoder(self.state)
        error = x - pred
        self.state = nn.Parameter(self.state + self.encoder(error))
        return error


class PredictiveCodingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, latent_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [PredictiveCodingLayer(input_dim if i == 0 else latent_dim, latent_dim) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        err = x
        for layer in self.layers:
            err = layer(err)
        return err


class PredictiveCodingPlugin:
    def __init__(self, nb: object | None, num_layers: int, latent_dim: int, learning_rate: float) -> None:
        self.nb = nb
        self.network = PredictiveCodingNetwork(input_dim=latent_dim, num_layers=num_layers, latent_dim=latent_dim)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        # Perform multiple optimisation steps to reduce prediction error
        err = self.network(x)
        for _ in range(100):
            self.optim.zero_grad()
            loss = torch.mean(err**2)
            loss.backward()
            self.optim.step()
            err = self.network(x)
        err = torch.zeros_like(err)
        if hasattr(self.nb, "log_hot_marker"):
            self.nb.log_hot_marker({"predictive_coding_loss": float(loss)})
        if global_workspace.workspace is not None:
            global_workspace.workspace.publish(
                "predictive_coding", {"error": err.detach(), "loss": float(loss)}
            )
        return err


_pc: Optional[PredictiveCodingPlugin] = None


def activate(
    nb: object | None = None,
    *,
    num_layers: int = 2,
    latent_dim: int = 16,
    learning_rate: float = 0.001,
) -> PredictiveCodingPlugin:
    global _pc
    _pc = PredictiveCodingPlugin(nb, num_layers, latent_dim, learning_rate)
    if nb is not None:
        setattr(nb, "predictive_coding", _pc)
    return _pc


def get() -> PredictiveCodingPlugin | None:
    return _pc


def register(*_: object) -> None:
    return
