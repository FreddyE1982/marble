"""Simple plugin loader for MARBLE."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Callable, Iterable, Type

import torch.nn as nn

from marble_core import LOSS_MODULES, NEURON_TYPES, SYNAPSE_TYPES

NeuronRegFunc = Callable[[str], None]
SynapseRegFunc = Callable[[str], None]
LossRegFunc = Callable[[str, Type[nn.Module]], None]


def register_neuron_type(name: str) -> None:
    """Register a custom neuron type if not already present."""
    if name not in NEURON_TYPES:
        NEURON_TYPES.append(name)


def register_synapse_type(name: str) -> None:
    """Register a custom synapse type if not already present."""
    if name not in SYNAPSE_TYPES:
        SYNAPSE_TYPES.append(name)


def register_loss_module(name: str, module: Type[nn.Module]) -> None:
    """Register a custom loss module if not already present."""
    if name not in LOSS_MODULES:
        LOSS_MODULES[name] = module


def load_plugins(dirs: Iterable[str] | str) -> None:
    """Load plugin modules from ``dirs``.

    Each module may define ``register`` with either two or three parameters.
    ``register(reg_neuron, reg_synapse)`` registers neuron and synapse types.
    ``register(reg_neuron, reg_synapse, reg_loss)`` additionally registers
    custom loss modules.
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
                    func = module.register
                    params = inspect.signature(func).parameters
                    if len(params) >= 3:
                        func(
                            register_neuron_type,
                            register_synapse_type,
                            register_loss_module,
                        )
                    else:
                        func(register_neuron_type, register_synapse_type)
