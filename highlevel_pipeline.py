from __future__ import annotations

import importlib
import inspect
import json
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
    """Build and execute sequential MARBLE operations."""

    DEFAULT_BIT_PARAMS = {
        "mixed": True,
        "max_vocab_size": None,
        "min_word_length": 4,
        "min_occurrence": 4,
    }

    DEFAULT_DATA_ARGS = {
        "data",
        "pairs",
        "train_examples",
        "validation_examples",
        "examples",
        "labeled_pairs",
        "unlabeled_inputs",
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
            min_occurrence=self.bit_dataset_params["min_occurrence"],
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
        current_marble = marble
        results: list[Any] = []
        for step in self.steps:
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
            # convert dataset arguments if requested
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
