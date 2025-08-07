"""Pipeline step plugin system.

This module defines the public interface for pipeline step plugins and
provides utilities to register and dynamically load third‑party plugins.
Plugins are normal Python classes implementing the :class:`PipelinePlugin`
interface which contains three lifecycle methods:

``initialise``
    Called once before execution.  Plugins may allocate resources here.  The
    selected ``device`` (CPU or GPU) is supplied so tensors can be moved to the
    appropriate target.
``execute``
    Run the step.  Results are returned to the pipeline and become the input to
    subsequent steps.  ``execute`` may accept the active MARBLE instance if
    required.
``teardown``
    Invoked after execution allowing plugins to release resources.

Third‑party packages may expose entry points under
``"marble.pipeline_plugins"`` or provide a ``register`` function inside a
Python file located in a configured directory.  ``load_pipeline_plugins`` will
locate these modules and populate the registry mapping identifiers to classes.
"""

from __future__ import annotations

import asyncio
from importlib import metadata, util
from pathlib import Path
from typing import Callable, Dict, Iterable, Type

import torch

from async_gradient_accumulator import AsyncGradientAccumulator


class PipelinePlugin:
    """Base class for pipeline step plugins.

    Subclasses may override :meth:`initialise`, :meth:`execute` and
    :meth:`teardown`.  ``initialise`` and ``execute`` receive the selected
    :class:`torch.device` so implementations can route tensors and operations to
    the proper hardware.  The active MARBLE instance is passed when available.
    """

    def __init__(self, **kwargs) -> None:  # pragma: no cover - simple storage
        self.params = kwargs

    def initialise(self, device: torch.device, marble=None) -> None:
        """Prepare the plugin for execution."""

    def execute(self, device: torch.device, marble=None):  # pragma: no cover - abstract
        """Perform the plugin's action and return the result."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Release any held resources."""


# Registry of available plugin classes keyed by identifier
PLUGIN_REGISTRY: Dict[str, Type[PipelinePlugin]] = {}


def register_plugin(name: str, plugin_cls: Type[PipelinePlugin]) -> None:
    """Register ``plugin_cls`` under ``name``.

    Existing registrations are overwritten, allowing users to replace default
    implementations.
    """

    PLUGIN_REGISTRY[name] = plugin_cls


def get_plugin(name: str) -> Type[PipelinePlugin]:
    """Return the plugin class registered as ``name``."""

    return PLUGIN_REGISTRY[name]


def load_pipeline_plugins(dirs: Iterable[str] | str | None = None) -> None:
    """Discover pipeline plugins from entry points or directories.

    Parameters
    ----------
    dirs:
        A path or iterable of paths to scan for modules defining a ``register``
        function.  Each ``register`` function receives :func:`register_plugin` as
        its argument.  If ``None`` no directories are scanned.

    Entry points exposed via the ``marble.pipeline_plugins`` group are loaded as
    well.  Each entry point should resolve to a class implementing
    :class:`PipelinePlugin` and will be registered under the entry point's name.
    """

    # Load entry points provided by installed packages
    try:
        entry_points = metadata.entry_points(group="marble.pipeline_plugins")
    except Exception:  # pragma: no cover - metadata behaviour varies
        entry_points = []
    for ep in entry_points:
        cls = ep.load()
        register_plugin(ep.name, cls)

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
                    module.register(register_plugin)


class ExportModelPlugin(PipelinePlugin):
    """Pipeline plugin that exports a trained model to disk.

    Parameters
    ----------
    path:
        Destination file path. The directory is created if necessary.
    fmt:
        Export format. ``"json"`` writes the core as JSON while ``"onnx"``
        uses :func:`marble_utils.export_core_to_onnx` and requires the
        :mod:`onnx` package. Both options operate on CPU or GPU depending on
        the currently selected device.
    """

    def __init__(self, path: str, fmt: str = "json") -> None:
        super().__init__(path=path, fmt=fmt)
        self.path = Path(path)
        self.fmt = fmt

    def initialise(
        self, device: torch.device, marble=None
    ) -> None:  # pragma: no cover - simple
        self.device = device
        self.marble = marble
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, device: torch.device, marble=None):
        if marble is None:
            raise ValueError("No model available for export")
        if hasattr(marble, "get_core"):
            core = marble.get_core()
        elif hasattr(marble, "core"):
            core = marble.core
        else:
            core = marble
        if self.fmt == "json":
            from marble_utils import core_to_json

            js = core_to_json(core)
            self.path.write_text(js, encoding="utf-8")
        elif self.fmt == "onnx":
            from marble_utils import export_core_to_onnx

            export_core_to_onnx(core, str(self.path))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported format: {self.fmt}")
        return str(self.path)

    def teardown(self) -> None:  # pragma: no cover - nothing to release
        pass


# Register built-in plugins
register_plugin("export_model", ExportModelPlugin)


class ServeModelPlugin(PipelinePlugin):
    """Start an HTTP server exposing a brain for inference.

    The plugin launches :class:`web_api.InferenceServer` on the selected device
    and returns a dictionary containing ``host``, ``port`` and the running
    server instance.  The server continues serving requests after the pipeline
    completes; callers are responsible for invoking ``server.stop()`` when
    finished.
    """

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        from config_loader import load_config

        cfg = load_config()
        defaults = cfg.get("mcp_server", {})
        host = host if host is not None else defaults.get("host", "localhost")
        port = port if port is not None else defaults.get("port", 5080)
        super().__init__(host=host, port=port)
        self.host = host
        self.port = port
        self.server = None

    def initialise(self, device: torch.device, marble=None) -> None:
        from web_api import InferenceServer

        if marble is None:
            raise ValueError("No model available for serving")
        brain = (
            marble.get_brain()
            if hasattr(marble, "get_brain")
            else getattr(marble, "brain", marble)
        )
        if hasattr(brain, "neuronenblitz") and hasattr(brain.neuronenblitz, "device"):
            brain.neuronenblitz.device = device
        self.server = InferenceServer(brain, host=self.host, port=self.port)

    def execute(self, device: torch.device, marble=None):
        self.server.start()
        return {"host": self.host, "port": self.port, "server": self.server}

    def teardown(self) -> None:
        # Server is intentionally left running for external use.
        pass


register_plugin("serve_model", ServeModelPlugin)


class MCPServeModelPlugin(PipelinePlugin):
    """Launch an MCPServer for MARBLE inference.

    This plugin mirrors :class:`ServeModelPlugin` but exposes the model via the
    MCP protocol. The server is started during :meth:`initialise` and remains
    running after the pipeline completes. The :meth:`execute` method simply
    returns connection information for clients.
    """

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        from config_loader import load_config

        cfg = load_config()
        defaults = cfg.get("mcp_server", {})
        host = host if host is not None else defaults.get("host", "localhost")
        port = port if port is not None else defaults.get("port", 5080)
        super().__init__(host=host, port=port)
        self.host = host
        self.port = port
        self.server = None

    def initialise(self, device: torch.device, marble=None) -> None:
        from mcp_server import MCPServer

        if marble is None:
            raise ValueError("No model available for serving")
        brain = (
            marble.get_brain()
            if hasattr(marble, "get_brain")
            else getattr(marble, "brain", marble)
        )
        if hasattr(brain, "neuronenblitz") and hasattr(brain.neuronenblitz, "device"):
            brain.neuronenblitz.device = device
        self.server = MCPServer(brain, host=self.host, port=self.port)
        self.server.start()

    def execute(self, device: torch.device, marble=None):
        return {"host": self.host, "port": self.port, "server": self.server}

    def teardown(self) -> None:
        # Server intentionally left running for external use.
        pass


register_plugin("serve_model_mcp", MCPServeModelPlugin)


class AsyncGradientAccumulationPlugin(PipelinePlugin):
    """Train a model using :class:`AsyncGradientAccumulator`.

    Parameters
    ----------
    model:
        PyTorch module to optimise.
    optimizer:
        Optimiser updating ``model`` parameters.
    loss_fn:
        Loss function applied to model outputs and targets.
    dataloader:
        Iterable yielding ``(inputs, targets)`` pairs.
    accumulation_steps:
        Number of micro-batches to accumulate before stepping the optimiser.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader,
        *,
        accumulation_steps: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=dataloader,
            accumulation_steps=accumulation_steps,
        )
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.accumulation_steps = accumulation_steps
        self.accumulator: AsyncGradientAccumulator | None = None

    def initialise(self, device: torch.device, marble=None) -> None:
        self.accumulator = AsyncGradientAccumulator(
            self.model,
            self.optimizer,
            self.loss_fn,
            accumulation_steps=self.accumulation_steps,
            device=device,
        )

    def execute(self, device: torch.device, marble=None):
        async def _run() -> None:
            assert self.accumulator is not None
            for inputs, targets in self.dataloader:
                await self.accumulator.add_batch(inputs, targets)
            await self.accumulator.flush()

        asyncio.run(_run())
        return self.model


register_plugin("async_gradient_accumulation", AsyncGradientAccumulationPlugin)
