from __future__ import annotations

import importlib
from typing import Any

import torch

from .base import RemoteTier
import pipeline_plugins


class MockRemoteTier(RemoteTier):
    """Local execution tier used for tests and examples.

    The tier pretends to be remote but simply executes the supplied step in the
    current process. It respects the requested device so tests can exercise the
    remote scheduling code paths without needing an actual service.
    """

    name = "mock"

    def connect(self) -> None:  # pragma: no cover - nothing to do
        pass

    def offload_core(self, core_bytes: bytes) -> bytes:  # pragma: no cover
        return core_bytes

    def run_step(self, step: dict, marble: Any | None, device: torch.device) -> Any:
        module_name = step.get("module")
        func_name = step.get("func")
        params = dict(step.get("params", {}))
        params.setdefault("device", device.type)
        if "plugin" in step:
            plugin_cls = pipeline_plugins.get_plugin(step["plugin"])
            plugin = plugin_cls(**params)
            plugin.initialise(device=device, marble=marble)
            try:
                return plugin.execute(device=device, marble=marble)
            finally:
                plugin.teardown()
        if func_name is None:
            raise ValueError("Step missing 'func'")
        module = importlib.import_module(module_name) if module_name else importlib.import_module("__main__")
        func = getattr(module, func_name)
        try:
            return func(marble, **params)
        except TypeError:
            return func(**params)

    def close(self) -> None:  # pragma: no cover - nothing to do
        pass


def get_remote_tier(address: str = "", **kwargs) -> MockRemoteTier:
    return MockRemoteTier(address)
