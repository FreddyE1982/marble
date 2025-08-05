from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
from typing import Dict, Iterable, Type

import torch

"""Plugin interface for MARBLE learning modules.

This module allows different learning algorithms to be swapped in and out at
runtime.  Each plugin is a subclass of :class:`LearningModule` and may be
registered either via the :func:`register_learning_module` function, by exposing
an entry point under ``marble.learning_plugins`` or by providing a ``register``
function inside a Python file located in a configured directory.

Plugins receive the active ``torch.device`` so implementations can run on CPU or
GPU seamlessly.  The optional ``marble`` argument provides access to the
currently active model for more advanced algorithms.
"""


class LearningModule:
    """Base class for swappable learning modules."""

    def __init__(self, **kwargs) -> None:  # pragma: no cover - storage only
        self.params = kwargs

    def initialise(self, device: torch.device, marble=None) -> None:
        """Prepare the module for training on ``device``."""

    def train_step(
        self, *args, device: torch.device, marble=None
    ):  # pragma: no cover - abstract
        """Execute a single training step and optionally return a loss."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Release any held resources."""


# Registry of learning module classes
LEARNING_MODULES: Dict[str, Type[LearningModule]] = {}


def register_learning_module(name: str, cls: Type[LearningModule]) -> None:
    """Register ``cls`` under ``name`` replacing any existing entry."""

    LEARNING_MODULES[name] = cls


def get_learning_module(name: str) -> Type[LearningModule]:
    """Return the learning module class registered as ``name``."""

    return LEARNING_MODULES[name]


def load_learning_plugins(dirs: Iterable[str] | str | None = None) -> None:
    """Discover learning module plugins from entry points or directories."""

    try:
        entry_points = metadata.entry_points(group="marble.learning_plugins")
    except Exception:  # pragma: no cover - metadata behaviour varies
        entry_points = []
    for ep in entry_points:
        cls = ep.load()
        register_learning_module(ep.name, cls)

    if dirs is None:
        return
    if isinstance(dirs, str):
        dirs = [dirs]

    for d in dirs:
        path = Path(d)
        if not path.is_dir():
            continue
        for file in path.glob("*.py"):
            spec = util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "register"):
                    module.register(register_learning_module)
