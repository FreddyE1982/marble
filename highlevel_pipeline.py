from __future__ import annotations

import importlib
import inspect
import json
import copy
from typing import Any, Callable, Iterable

from torch.utils.data import Dataset

import marble_interface
from bit_tensor_dataset import BitTensorDataset


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
    ) -> None:
        self.steps: list[dict] = steps or []
        self.use_bit_dataset = use_bit_dataset
        self.bit_dataset_params = self.DEFAULT_BIT_PARAMS.copy()
        self.data_args = set(data_args or self.DEFAULT_DATA_ARGS)
        if bit_dataset_params:
            self.bit_dataset_params.update(bit_dataset_params)

    def set_bit_dataset_params(self, **params: Any) -> None:
        """Update default parameters for :class:`BitTensorDataset`."""
        self.bit_dataset_params.update(params)

    def register_data_args(self, *names: str) -> None:
        """Treat ``names`` as dataset parameters for automatic conversion."""
        self.data_args.update(names)

    def __getattr__(self, name: str) -> Callable | _ModuleWrapper:
        if hasattr(marble_interface, name):

            def wrapper(**params: Any) -> "HighLevelPipeline":
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
            self.steps.append({"callable": func, "params": params or {}})
        else:
            self.steps.append({"func": func, "module": module, "params": params or {}})
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

    def duplicate(self) -> "HighLevelPipeline":
        """Return a deep copy of this pipeline."""
        return HighLevelPipeline(
            steps=copy.deepcopy(self.steps),
            use_bit_dataset=self.use_bit_dataset,
            bit_dataset_params=self.bit_dataset_params.copy(),
            data_args=self.data_args.copy(),
        )

    def _execute_steps(
        self, steps: list[dict], marble: Any | None
    ) -> tuple[Any | None, list[Any]]:
        """Internal helper executing ``steps`` sequentially."""
        current_marble = marble
        results: list[Any] = []
        for step in steps:
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
            found = self._extract_marble(result)
            if found is not None:
                current_marble = found
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
            mixed=self.bit_dataset_params["mixed"],
            max_vocab_size=self.bit_dataset_params["max_vocab_size"],
            min_word_length=self.bit_dataset_params["min_word_length"],
            max_word_length=self.bit_dataset_params["max_word_length"],
            min_occurrence=self.bit_dataset_params["min_occurrence"],
            vocab=self.bit_dataset_params.get("vocab"),
            device=self.bit_dataset_params["device"],
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

    def execute(self, marble: Any | None = None) -> tuple[Any | None, list[Any]]:
        return self._execute_steps(self.steps, marble)

    def run_step(self, index: int, marble: Any | None = None) -> tuple[Any | None, Any]:
        """Execute a single step at ``index`` and return the result."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        marble, results = self._execute_steps([self.steps[index]], marble)
        return marble, results[0]

    def execute_until(self, index: int, marble: Any | None = None) -> tuple[Any | None, list[Any]]:
        """Execute pipeline steps up to ``index`` (inclusive)."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        return self._execute_steps(self.steps[: index + 1], marble)

    def save_json(self, path: str) -> None:
        for step in self.steps:
            if "callable" in step:
                raise ValueError("Cannot serialise pipelines containing callables")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=2)

    @classmethod
    def load_json(cls, file_obj) -> "HighLevelPipeline":
        steps = json.load(file_obj)
        return cls(steps=steps)
