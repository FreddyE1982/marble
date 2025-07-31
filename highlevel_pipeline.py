from __future__ import annotations

import importlib
import inspect
import json
from typing import Any, Callable

import marble_interface


class HighLevelPipeline:
    """Build and execute sequential MARBLE operations."""

    def __init__(self, steps: list[dict] | None = None) -> None:
        self.steps: list[dict] = steps or []

    def __getattr__(self, name: str) -> Callable:
        if hasattr(marble_interface, name):
            def wrapper(**params):
                self.add_step(name, module="marble_interface", params=params)
                return self
            return wrapper
        raise AttributeError(name)

    def add_step(
        self,
        func: str | Callable,
        *,
        module: str | None = None,
        params: dict | None = None,
    ) -> 'HighLevelPipeline':
        if callable(func):
            module = module or func.__module__
            func = func.__name__
        self.steps.append({"func": func, "module": module, "params": params or {}})
        return self

    def execute(self, marble: Any | None = None) -> tuple[Any | None, list[Any]]:
        current_marble = marble
        results: list[Any] = []
        for step in self.steps:
            module_name = step.get("module")
            func_name = step["func"]
            params = step.get("params", {})
            module = importlib.import_module(module_name) if module_name else marble_interface
            if not hasattr(module, func_name):
                raise ValueError(f"Unknown function: {func_name}")
            func = getattr(module, func_name)
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
            if isinstance(result, marble_interface.MARBLE):
                current_marble = result
            results.append(result)
        return current_marble, results

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=2)

    @classmethod
    def load_json(cls, file_obj) -> "HighLevelPipeline":
        steps = json.load(file_obj)
        return cls(steps=steps)
