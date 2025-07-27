"""Simple plugin loader for MARBLE."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Iterable

from marble_core import NEURON_TYPES, SYNAPSE_TYPES


NeuronRegFunc = Callable[[str], None]
SynapseRegFunc = Callable[[str], None]


def register_neuron_type(name: str) -> None:
    """Register a custom neuron type if not already present."""
    if name not in NEURON_TYPES:
        NEURON_TYPES.append(name)


def register_synapse_type(name: str) -> None:
    """Register a custom synapse type if not already present."""
    if name not in SYNAPSE_TYPES:
        SYNAPSE_TYPES.append(name)


def load_plugins(dirs: Iterable[str] | str) -> None:
    """Load plugin modules from ``dirs``.

    Each module may define ``register(register_neuron, register_synapse)`` which
    is called with the registration callbacks.
    """
    if isinstance(dirs, str):
        dirs = [dirs]
    for d in dirs:
        path = Path(d)
        if not path.is_dir():
            continue
        for file in path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register"):
                    module.register(register_neuron_type, register_synapse_type)
