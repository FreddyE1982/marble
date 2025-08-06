from __future__ import annotations

import importlib
import inspect
import json
import copy
from typing import Any, Callable, Iterable
import asyncio
import os
import pickle

import torch
import psutil
from dotdict import DotDict
from config_loader import load_config
from marble_base import MetricsVisualizer

from torch.utils.data import Dataset

import marble_interface
from bit_tensor_dataset import BitTensorDataset, bytes_to_object, tensors_to_bytes
from marble_neuronenblitz import Neuronenblitz


class _ModuleWrapper:
    """Helper that exposes module functions as pipeline steps."""

    def __init__(self, pipeline: "HighLevelPipeline", module: Any) -> None:
        self._pipeline = pipeline
        self._module = module

    def __getattr__(self, func_name: str) -> Callable:
        if hasattr(self._module, func_name):
            attr = getattr(self._module, func_name)
            if inspect.ismodule(attr):
                return _ModuleWrapper(self._pipeline, attr)

            def wrapper(**params: Any) -> "HighLevelPipeline":
                self._pipeline.add_step(
                    func_name,
                    module=self._module.__name__,
                    params=params,
                )
                return self._pipeline

            return wrapper

        try:
            submodule = importlib.import_module(f"{self._module.__name__}.{func_name}")
        except ModuleNotFoundError as exc:  # pragma: no cover - handled in tests
            raise AttributeError(func_name) from exc

        return _ModuleWrapper(self._pipeline, submodule)


class HighLevelPipeline:
    """Build and execute sequential MARBLE operations.

    The pipeline automatically wraps dataset-like arguments in
    :class:`BitTensorDataset` using ``mixed`` mode, no vocabulary size limit and
    a minimum and maximum word length of ``4`` and ``8`` with automatic device
    placement. A custom vocabulary may be
    supplied so multiple datasets share the same token mapping. Functions from
    :mod:`marble_interface` as well as any other module within the repository
    can be appended dynamically via attribute access. There is no imposed limit
    on pipeline length so users may chain together an unlimited number of
    operations. Steps can be reordered or removed, making it possible to
    experiment with any MARBLE feature or option in a single workflow simply by
    adding steps in the desired order.
    """

    DEFAULT_BIT_PARAMS = {
        "mixed": True,
        "max_vocab_size": None,
        "min_word_length": 4,
        "max_word_length": 8,
        "min_occurrence": 4,
        "start_id": 256,
        "vocab": None,
        "device": None,
        "compress": False,
    }

    DEFAULT_DATA_ARGS = {
        "data",
        "pairs",
        "train_examples",
        "validation_examples",
        "examples",
        "labeled_pairs",
        "unlabeled_inputs",
        "features",
    }

    def __init__(
        self,
        steps: list[dict] | None = None,
        *,
        use_bit_dataset: bool = True,
        bit_dataset_params: dict | None = None,
        data_args: Iterable[str] | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        dataset_version: str | None = None,
        async_enabled: bool | None = None,
    ) -> None:
        self.steps: list[dict] = []
        for step in steps or []:
            new = step.copy()
            new["params"] = DotDict(step.get("params", {}))
            self.steps.append(DotDict(new))
        self.use_bit_dataset = use_bit_dataset
        self.bit_dataset_params = DotDict(self.DEFAULT_BIT_PARAMS.copy())
        self.data_args = set(data_args or self.DEFAULT_DATA_ARGS)
        self.config = DotDict(load_config(config_path))
        if bit_dataset_params:
            self.bit_dataset_params.update(bit_dataset_params)
        pipeline_cfg = self.config.get("pipeline", {})
        if cache_dir is None:
            cfg_cache = pipeline_cfg.get("cache_dir")
            if cfg_cache is None:
                cache_dir = (
                    "pipeline_cache_gpu" if torch.cuda.is_available() else "pipeline_cache_cpu"
                )
            else:
                cache_dir = cfg_cache
        self.cache_dir = cache_dir
        self.dataset_version = dataset_version
        self.async_enabled = (
            async_enabled if async_enabled is not None else pipeline_cfg.get("async_enabled", False)
        )
        self.default_memory_limit_mb = pipeline_cfg.get("default_step_memory_limit_mb")

    def set_bit_dataset_params(self, **params: Any) -> None:
        """Update default parameters for :class:`BitTensorDataset`."""
        self.bit_dataset_params.update(params)

    def register_data_args(self, *names: str) -> None:
        """Treat ``names`` as dataset parameters for automatic conversion."""
        self.data_args.update(names)

    def __getattr__(self, name: str) -> Callable | _ModuleWrapper:
        if hasattr(marble_interface, name):

            def wrapper(**params: Any) -> "HighLevelPipeline":
                if name in {"new_marble_system", "configure_marble_system"}:
                    params.setdefault("config", self.config.to_dict())
                self.add_step(name, module="marble_interface", params=params)
                return self

            return wrapper

        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise AttributeError(name) from exc

        return _ModuleWrapper(self, module)

    def add_step(
        self,
        func: str | Callable,
        *,
        module: str | None = None,
        params: dict | None = None,
    ) -> "HighLevelPipeline":
        """Append ``func`` as a pipeline step.

        ``func`` may be a callable or the name of a function. When a callable is
        supplied it is stored directly and executed without an import. Such steps
        cannot be serialised with :meth:`save_json`.
        """
        if callable(func):
            step = {"callable": func, "params": DotDict(params or {})}
        else:
            step = {"func": func, "module": module, "params": DotDict(params or {})}
        self.steps.append(DotDict(step))
        return self

    def remove_step(self, index: int) -> None:
        """Remove a step at ``index``."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        self.steps.pop(index)

    def move_step(self, old_index: int, new_index: int) -> None:
        """Reorder a step from ``old_index`` to ``new_index``."""
        if old_index < 0 or old_index >= len(self.steps):
            raise IndexError("old_index out of range")
        if new_index < 0 or new_index >= len(self.steps):
            raise IndexError("new_index out of range")
        step = self.steps.pop(old_index)
        self.steps.insert(new_index, step)

    def insert_step(
        self,
        index: int,
        func: str | Callable,
        *,
        module: str | None = None,
        params: dict | None = None,
    ) -> None:
        """Insert ``func`` as a step at ``index``.

        The same rules as :meth:`add_step` apply for callables versus named
        functions. ``index`` may point to the end of the list to append.
        """

        if index < 0 or index > len(self.steps):
            raise IndexError("index out of range")
        if callable(func):
            step = {"callable": func, "params": DotDict(params or {})}
        else:
            step = {"func": func, "module": module, "params": DotDict(params or {})}
        self.steps.insert(index, DotDict(step))

    def replace_step(
        self,
        index: int,
        func: str | Callable,
        *,
        module: str | None = None,
        params: dict | None = None,
    ) -> None:
        """Replace the step at ``index`` with ``func``."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        if callable(func):
            step = {"callable": func, "params": DotDict(params or {})}
        else:
            step = {"func": func, "module": module, "params": DotDict(params or {})}
        self.steps[index] = DotDict(step)

    def update_step_params(self, index: int, **params: Any) -> None:
        """Update stored parameters for the step at ``index``."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        self.steps[index].setdefault("params", DotDict()).update(params)

    def duplicate(self) -> "HighLevelPipeline":
        """Return a deep copy of this pipeline."""
        return HighLevelPipeline(
            steps=copy.deepcopy(self.steps),
            use_bit_dataset=self.use_bit_dataset,
            bit_dataset_params=self.bit_dataset_params.copy(),
            data_args=self.data_args.copy(),
            cache_dir=self.cache_dir,
            dataset_version=self.dataset_version,
            async_enabled=self.async_enabled,
        )

    def get_step(self, index: int) -> dict:
        """Return the step dictionary at ``index``."""

        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return self.steps[index]

    def list_steps(self) -> list[str]:
        """Return a list of step names for introspection."""

        names = []
        for step in self.steps:
            if "callable" in step:
                names.append(step["callable"].__name__)
            else:
                mod = step.get("module", "marble_interface")
                names.append(f"{mod}.{step['func']}")
        return names

    def _is_dataset_step(self, step: dict) -> bool:
        """Return ``True`` if ``step`` appears to produce a dataset."""

        name = ""
        if "func" in step:
            name = step["func"]
        elif "callable" in step:
            name = step["callable"].__name__
        return "dataset" in name.lower()

    def _train_neuronenblitz(
        self, nb: Neuronenblitz, dataset, device: torch.device, epochs: int = 1
    ) -> None:
        """Train ``nb`` on ``dataset`` which may be a list of batches."""
        
        def _decode(t: torch.Tensor):
            if isinstance(t, torch.Tensor) and t.dtype in {torch.uint8, torch.int32, torch.int64}:
                try:
                    return bytes_to_object(tensors_to_bytes(t.cpu()))
                except Exception:
                    return t
            return t

        if (
            isinstance(dataset, list)
            and dataset
            and isinstance(dataset[0], dict)
            and "inputs" in dataset[0]
        ):
            for batch in dataset:
                inputs = batch.get("inputs")
                targets = batch.get("targets")
                if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                    continue
                pairs = []
                for inp_t, tgt_t in zip(inputs, targets):
                    inp = _decode(inp_t)
                    tgt = _decode(tgt_t)
                    if isinstance(inp, torch.Tensor):
                        inp = inp.to(device)
                    if isinstance(tgt, torch.Tensor):
                        tgt = tgt.to(device)
                    pairs.append((inp, tgt))
                if pairs:
                    nb.train(pairs, epochs=epochs)
            return

        prepared = []
        for item in dataset:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            inp, tgt = item[0], item[1]
            inp = _decode(inp)
            tgt = _decode(tgt)
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.to(device)
            prepared.append((inp, tgt))
        if prepared:
            nb.train(prepared, epochs=epochs)

    def _execute_steps(
        self,
        steps: list[dict],
        marble: Any | None,
        metrics_visualizer: MetricsVisualizer | None = None,
    ) -> tuple[Any | None, list[Any]]:
        """Internal helper executing ``steps`` sequentially."""
        current_marble = marble
        results: list[Any] = []
        for idx, step in enumerate(steps):
            global_index = self.steps.index(step)
            cache_hit = False
            cache_path = None
            if self.cache_dir:
                cache_path = self._cache_path(global_index, step)
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        result = pickle.load(f)
                    found = self._extract_marble(result)
                    if found is not None:
                        current_marble = found
                    results.append(result)
                    cache_hit = True
            if cache_hit:
                if metrics_visualizer:
                    metrics_visualizer.update({"cache_hit": 1})
                continue
            if "callable" in step:
                func = step["callable"]
                params = step.get("params", {})
            else:
                module_name = step.get("module")
                func_name = step["func"]
                params = step.get("params", {})
                module = (
                    importlib.import_module(module_name)
                    if module_name
                    else marble_interface
                )
                if not hasattr(module, func_name):
                    raise ValueError(f"Unknown function: {func_name}")
                func = getattr(module, func_name)
            new_params = {}
            for k, v in params.items():
                if k in self.data_args:
                    new_params[k] = self._maybe_bit_dataset(v)
                else:
                    new_params[k] = v
            params = new_params
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            process = psutil.Process()
            cpu_before = process.memory_info().rss
            gpu_before = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
            sig = inspect.signature(func)
            kwargs = {}
            for name, p in sig.parameters.items():
                if name == "marble":
                    kwargs[name] = current_marble
                elif name in params:
                    kwargs[name] = params[name]
                elif p.default is not inspect.Parameter.empty:
                    kwargs[name] = p.default
                else:
                    raise ValueError(f"Missing parameter: {name}")
            result = func(**kwargs)
            if metrics_visualizer:
                metrics_visualizer.update({"cache_miss": 1})
            if self.cache_dir and cache_path:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            if hasattr(result, "next_batch") and hasattr(result, "is_finished"):
                async def _drain(step):
                    batches = []
                    async for batch in step:
                        batches.append(batch)
                    step.close()
                    return batches

                result = asyncio.run(_drain(result))
            found = self._extract_marble(result)
            if found is not None:
                current_marble = found
            if (
                isinstance(current_marble, Neuronenblitz)
                and self._is_dataset_step(step)
                and isinstance(result, list)
            ):
                epochs = int(step.get("params", {}).get("epochs", 1))
                self._train_neuronenblitz(current_marble, result, device, epochs=epochs)
            cpu_after = process.memory_info().rss
            gpu_after = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
            usage = max(cpu_after - cpu_before, gpu_after - gpu_before)
            limit_mb = step.get("memory_limit_mb", self.default_memory_limit_mb)
            if limit_mb is not None and usage > limit_mb * 1024 * 1024:
                step_name = step.get("name") or step.get("func") or "callable"
                raise MemoryError(
                    f"Step '{step_name}' exceeded memory limit of {limit_mb} MB"
                )
            results.append(result)
        return current_marble, results

    async def _execute_steps_async(
        self,
        steps: list[dict],
        marble: Any | None,
        metrics_visualizer: MetricsVisualizer | None = None,
    ) -> tuple[Any | None, list[Any]]:
        current_marble = marble
        results: list[Any] = []
        loop = asyncio.get_event_loop()
        for idx, step in enumerate(steps):
            global_index = self.steps.index(step)
            cache_hit = False
            cache_path = None
            if self.cache_dir:
                cache_path = self._cache_path(global_index, step)
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        result = pickle.load(f)
                    found = self._extract_marble(result)
                    if found is not None:
                        current_marble = found
                    results.append(result)
                    cache_hit = True
            if cache_hit:
                if metrics_visualizer:
                    metrics_visualizer.update({"cache_hit": 1})
                continue
            if "callable" in step:
                func = step["callable"]
                params = step.get("params", {})
            else:
                module_name = step.get("module")
                func_name = step["func"]
                params = step.get("params", {})
                module = (
                    importlib.import_module(module_name)
                    if module_name
                    else marble_interface
                )
                if not hasattr(module, func_name):
                    raise ValueError(f"Unknown function: {func_name}")
                func = getattr(module, func_name)
            new_params = {}
            for k, v in params.items():
                if k in self.data_args:
                    new_params[k] = self._maybe_bit_dataset(v)
                else:
                    new_params[k] = v
            params = new_params
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            process = psutil.Process()
            cpu_before = process.memory_info().rss
            gpu_before = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
            sig = inspect.signature(func)
            kwargs = {}
            for name, p in sig.parameters.items():
                if name == "marble":
                    kwargs[name] = current_marble
                elif name in params:
                    kwargs[name] = params[name]
                elif p.default is not inspect.Parameter.empty:
                    kwargs[name] = p.default
                else:
                    raise ValueError(f"Missing parameter: {name}")
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = await loop.run_in_executor(None, lambda: func(**kwargs))
            if metrics_visualizer:
                metrics_visualizer.update({"cache_miss": 1})
            if self.cache_dir and cache_path:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            if hasattr(result, "next_batch") and hasattr(result, "is_finished"):
                batches = []
                async for batch in result:
                    batches.append(batch)
                result.close()
                result = batches
            found = self._extract_marble(result)
            if found is not None:
                current_marble = found
            if (
                isinstance(current_marble, Neuronenblitz)
                and self._is_dataset_step(step)
                and isinstance(result, list)
            ):
                epochs = int(step.get("params", {}).get("epochs", 1))
                await loop.run_in_executor(
                    None,
                    lambda: self._train_neuronenblitz(
                        current_marble, result, device, epochs=epochs
                    ),
                )
            cpu_after = process.memory_info().rss
            gpu_after = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
            usage = max(cpu_after - cpu_before, gpu_after - gpu_before)
            limit_mb = step.get("memory_limit_mb", self.default_memory_limit_mb)
            if limit_mb is not None and usage > limit_mb * 1024 * 1024:
                step_name = step.get("name") or step.get("func") or "callable"
                raise MemoryError(
                    f"Step '{step_name}' exceeded memory limit of {limit_mb} MB"
                )
            results.append(result)
        return current_marble, results

    def describe(self) -> str:
        """Return a human readable string describing all steps."""
        lines: list[str] = []
        for i, step in enumerate(self.steps):
            if "callable" in step:
                name = step["callable"].__name__
            else:
                module = step.get("module", "marble_interface")
                name = f"{module}.{step['func']}"
            lines.append(f"{i}: {name} params={step.get('params', {})}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.describe()

    def _maybe_bit_dataset(self, obj: Any) -> Any:
        if not self.use_bit_dataset:
            return obj
        if isinstance(obj, BitTensorDataset):
            return obj
        if isinstance(obj, Dataset):
            try:
                items = [obj[i] for i in range(len(obj))]
            except Exception:  # pragma: no cover - handle iterables
                items = list(obj)
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
            items = list(obj)
        else:
            return obj
        if not items:
            return obj
        if isinstance(items[0], tuple) and len(items[0]) == 2:
            pairs = items
        else:
            pairs = [(x, x) for x in items]
        return BitTensorDataset(
            pairs,
            use_vocab=True,
            mixed=self.bit_dataset_params.mixed,
            max_vocab_size=self.bit_dataset_params.max_vocab_size,
            min_word_length=self.bit_dataset_params.min_word_length,
            max_word_length=self.bit_dataset_params.max_word_length,
            min_occurrence=self.bit_dataset_params.min_occurrence,
            start_id=self.bit_dataset_params.get("start_id", 256),
            vocab=self.bit_dataset_params.get("vocab"),
            device=self.bit_dataset_params.device,
            compress=self.bit_dataset_params.get("compress", False),
        )

    def _extract_marble(self, obj: Any) -> marble_interface.MARBLE | None:
        """Return the first :class:`MARBLE` instance found in ``obj``."""
        if isinstance(obj, marble_interface.MARBLE):
            return obj
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                m = self._extract_marble(item)
                if m is not None:
                    return m
        if isinstance(obj, dict):
            for val in obj.values():
                m = self._extract_marble(val)
                if m is not None:
                    return m
        return None

    def execute(
        self,
        marble: Any | None = None,
        *,
        metrics_visualizer: MetricsVisualizer | None = None,
    ) -> tuple[Any | None, list[Any]]:
        if self.async_enabled:
            return asyncio.run(
                self._execute_steps_async(self.steps, marble, metrics_visualizer)
            )
        return self._execute_steps(self.steps, marble, metrics_visualizer)

    async def execute_async(
        self,
        marble: Any | None = None,
        *,
        metrics_visualizer: MetricsVisualizer | None = None,
    ) -> tuple[Any | None, list[Any]]:
        return await self._execute_steps_async(self.steps, marble, metrics_visualizer)

    def execute_stream(self, marble: Any | None = None):
        """Yield ``(marble, result)`` tuples after each step executes."""
        current_marble = marble
        for step in self.steps:
            current_marble, result = self._execute_steps([step], current_marble)
            yield current_marble, result[0]

    def run_step(self, index: int, marble: Any | None = None) -> tuple[Any | None, Any]:
        """Execute a single step at ``index`` and return the result."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        marble, results = self._execute_steps([self.steps[index]], marble)
        return marble, results[0]

    async def run_step_async(self, index: int, marble: Any | None = None) -> tuple[Any | None, Any]:
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        marble, results = await self._execute_steps_async([self.steps[index]], marble)
        return marble, results[0]

    def execute_until(
        self, index: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        """Execute pipeline steps up to ``index`` (inclusive)."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return self._execute_steps(self.steps[: index + 1], marble)

    async def execute_until_async(
        self, index: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return await self._execute_steps_async(self.steps[: index + 1], marble)

    def execute_from(
        self, index: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        """Execute pipeline steps starting at ``index`` until the end."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return self._execute_steps(self.steps[index:], marble)

    async def execute_from_async(
        self, index: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return await self._execute_steps_async(self.steps[index:], marble)

    def execute_range(
        self, start: int, end: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        """Execute steps from ``start`` to ``end`` inclusive."""

        if start < 0 or end >= len(self.steps) or start > end:
            raise IndexError("invalid range")
        return self._execute_steps(self.steps[start : end + 1], marble)

    async def execute_range_async(
        self, start: int, end: int, marble: Any | None = None
    ) -> tuple[Any | None, list[Any]]:
        if start < 0 or end >= len(self.steps) or start > end:
            raise IndexError("invalid range")
        return await self._execute_steps_async(self.steps[start : end + 1], marble)

    def save_json(self, path: str) -> None:
        for step in self.steps:
            if "callable" in step:
                raise ValueError("Cannot serialise pipelines containing callables")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=2)

    def to_json(self) -> str:
        """Return a JSON string representing this pipeline."""
        for step in self.steps:
            if "callable" in step:
                raise ValueError("Cannot serialise pipelines containing callables")
        return json.dumps(self.steps, indent=2)

    @classmethod
    def load_json(cls, file_obj) -> "HighLevelPipeline":
        steps = json.load(file_obj)
        return cls(steps=steps)

    @classmethod
    def from_json(cls, json_str: str) -> "HighLevelPipeline":
        """Construct a pipeline from a JSON string."""
        steps = json.loads(json_str)
        return cls(steps=steps)

    # ------------------------------------------------------------------
    # Checkpointing helpers
    def save_checkpoint(self, path: str) -> None:
        data = {
            "steps": self.steps,
            "bit_dataset_params": self.bit_dataset_params.to_dict(),
            "use_bit_dataset": self.use_bit_dataset,
            "data_args": list(self.data_args),
            "config": self.config.to_dict(),
            "dataset_version": self.dataset_version,
            "cache_dir": self.cache_dir,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "HighLevelPipeline":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            steps=data.get("steps"),
            use_bit_dataset=data.get("use_bit_dataset", True),
            bit_dataset_params=data.get("bit_dataset_params"),
            data_args=data.get("data_args"),
            cache_dir=data.get("cache_dir"),
            dataset_version=data.get("dataset_version"),
        )
        obj.config = DotDict(data.get("config", {}))
        return obj

    def clear_steps(self) -> None:
        """Remove all steps from the pipeline."""

        self.steps.clear()

    def summary(self) -> dict[str, Any]:
        """Return a dictionary summarising the pipeline configuration."""

        return {
            "num_steps": len(self.steps),
            "use_bit_dataset": self.use_bit_dataset,
            "bit_dataset_params": self.bit_dataset_params.to_dict(),
            "config": self.config.to_dict(),
            "dataset_version": self.dataset_version,
        }

    # ------------------------------------------------------------------
    # Caching helpers
    def _cache_path(self, index: int, step: dict) -> str:
        assert self.cache_dir is not None
        name = step.get("func") or getattr(step.get("callable"), "__name__", "callable")
        return os.path.join(self.cache_dir, f"{index}_{name}.pkl")

    def clear_cache(self) -> None:
        if not self.cache_dir:
            return
        for f in os.listdir(self.cache_dir):
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.cache_dir, f))
                except FileNotFoundError:
                    pass
