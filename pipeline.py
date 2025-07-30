from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

import marble_interface


class Pipeline:
    """Sequence of function calls executable with an optional MARBLE instance."""

    def __init__(self, steps: list[dict] | None = None) -> None:
        self.steps: list[dict] = steps or []

    def add_step(self, func: str, *, module: str | None = None, params: dict | None = None) -> None:
        self.steps.append({"func": func, "module": module, "params": params or {}})

    def remove_step(self, index: int) -> None:
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        self.steps.pop(index)

    def move_step(self, old_index: int, new_index: int) -> None:
        if old_index < 0 or old_index >= len(self.steps):
            raise IndexError("old_index out of range")
        if new_index < 0 or new_index >= len(self.steps):
            raise IndexError("new_index out of range")
        step = self.steps.pop(old_index)
        self.steps.insert(new_index, step)

    def execute(self, marble: Any | None = None) -> list[Any]:
        results: list[Any] = []
        for step in self.steps:
            module_name = step.get("module")
            func_name = step["func"]
            params = step.get("params", {})
            results.append(self._execute_function(module_name, func_name, marble, params))
        return results

    def _execute_function(self, module_name: str | None, func_name: str, marble: Any, params: dict) -> Any:
        module = importlib.import_module(module_name) if module_name else marble_interface
        if not hasattr(module, func_name):
            raise ValueError(f"Unknown function: {func_name}")
        func = getattr(module, func_name)
        sig = inspect.signature(func)
        kwargs = {}
        for name, p in sig.parameters.items():
            if name == "marble" and marble is not None:
                kwargs[name] = marble
                continue
            if name in params:
                kwargs[name] = params[name]
            elif p.default is not inspect.Parameter.empty:
                kwargs[name] = p.default
            else:
                raise ValueError(f"Missing parameter: {name}")
        return func(**kwargs)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=2)

    @classmethod
    def load_json(cls, file_obj) -> "Pipeline":
        steps = json.load(file_obj)
        return cls(steps=steps)
