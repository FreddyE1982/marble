"""Example pipeline plugin that scales a tensor by a constant factor."""

from __future__ import annotations

import torch

from pipeline_plugins import PipelinePlugin, register_plugin


class DoubleStep(PipelinePlugin):
    """Multiply a tensor of ones by ``factor`` on the selected device."""

    def __init__(self, factor: float = 2.0) -> None:
        super().__init__(factor=factor)
        self.factor = factor
        self.device: torch.device | None = None

    def initialise(self, device: torch.device, marble=None) -> None:
        self.device = device

    def execute(self, device: torch.device, marble=None):
        tensor = torch.ones(1, device=device)
        return tensor * self.factor

    def teardown(self) -> None:
        self.device = None


def register(register_func) -> None:
    register_func("double_step", DoubleStep)
