"""Utility for loading remote hardware plugins."""

from __future__ import annotations

import importlib

from .base import RemoteTier


def load_remote_tier_plugin(module_name: str, **kwargs) -> RemoteTier:
    """Load ``module_name`` and return a ``RemoteTier`` instance."""
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_remote_tier"):
        raise ImportError(f"Plugin {module_name} lacks get_remote_tier()")
    tier = module.get_remote_tier(**kwargs)
    if not isinstance(tier, RemoteTier):
        raise TypeError("get_remote_tier did not return RemoteTier")
    return tier
