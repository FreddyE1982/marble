from __future__ import annotations

"""Registry and base classes for external tool plugins."""

from importlib import metadata, util
from pathlib import Path
from typing import Dict, Iterable, Type, Union

import torch

from pipeline_plugins import PipelinePlugin


class ToolPlugin(PipelinePlugin):
    """Base class for tool plugins used by :class:`ToolManagerPlugin`.

    Subclasses implement :meth:`can_handle` to declare whether a query should be
    routed to the tool and override :meth:`execute` to perform the actual work.
    The :meth:`execute` method must accept a ``query`` keyword argument so that
    the manager can supply the user's request.  Implementations are expected to
    operate on CPU or GPU depending on ``device`` but may ignore it when the
    underlying tool does not benefit from acceleration.
    """

    def can_handle(self, query: str) -> bool:  # pragma: no cover - interface
        """Return ``True`` if this tool can process ``query``."""
        raise NotImplementedError

    def execute(self, device: torch.device, marble=None, query: str = ""):
        raise NotImplementedError


TOOL_REGISTRY: Dict[str, Type[ToolPlugin]] = {}


def register_tool(name: str, plugin_cls: Type[ToolPlugin]) -> None:
    """Register ``plugin_cls`` under ``name`` for later lookup."""

    TOOL_REGISTRY[name] = plugin_cls


def get_tool(name: str) -> Type[ToolPlugin]:
    """Return the tool class previously registered as ``name``."""

    return TOOL_REGISTRY[name]


def load_tool_plugins(dirs: Iterable[Union[str, Path]] | str | Path | None = None) -> None:
    """Discover tool plugins from entry points or directories.

    Parameters
    ----------
    dirs:
        Optional path or iterable of paths to scan for Python files defining a
        ``register`` function.  Each ``register`` function receives
        :func:`register_tool` as its only argument and should use it to register
        one or more tool classes.  Entry points exposed under the
        ``"marble.tool_plugins"`` group are loaded as well.
    """

    try:
        entry_points = metadata.entry_points(group="marble.tool_plugins")
    except Exception:  # pragma: no cover - metadata behaviour varies
        entry_points = []
    for ep in entry_points:
        cls = ep.load()
        register_tool(ep.name, cls)

    if dirs is None:
        return
    if isinstance(dirs, (str, Path)):
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
                    module.register(register_tool)
