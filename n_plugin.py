"""Neuronenblitz plugin management."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict, Optional

# Active plugin modules keyed by name
_active_plugins: Dict[str, ModuleType] = {}
# Currently registered Neuronenblitz instance
_nb_instance: Optional[object] = None


def register(nb: object) -> None:
    """Register a ``Neuronenblitz`` instance for plugin access."""
    global _nb_instance
    _nb_instance = nb
    for module in _active_plugins.values():
        if hasattr(module, "activate"):
            module.activate(nb)


def activate(name: str) -> ModuleType:
    """Import and activate plugin ``name``."""
    module = import_module(name)
    _active_plugins[name] = module
    if _nb_instance is not None and hasattr(module, "activate"):
        module.activate(_nb_instance)
    return module
