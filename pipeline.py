from __future__ import annotations

import asyncio
import difflib
import importlib
import inspect
import json
import time
from typing import Any, Callable

import networkx as nx
import torch

import global_workspace
import marble_interface
import pipeline_plugins
from dataset_loader import wait_for_prefetch
from marble_base import MetricsVisualizer
from marble_core import benchmark_message_passing


class Pipeline:
    """Sequence of function calls executable with an optional MARBLE instance."""

    def __init__(self, steps: list[dict] | None = None) -> None:
        self.steps: list[dict] = steps or []
        self._summaries: list[dict] = []
        self._benchmarks: list[dict] = []

    def add_step(
        self,
        func: str | None,
        *,
        module: str | None = None,
        params: dict | None = None,
        name: str | None = None,
        depends_on: list[str] | None = None,
        plugin: str | None = None,
    ) -> None:
        """Add a step to the pipeline.

        Each step can declare a unique ``name`` used by other steps to express
        dependencies via ``depends_on``.  Steps are automatically reordered
        according to these dependencies when the pipeline executes.  If ``name``
        is omitted it defaults to ``func``/``plugin`` with an index suffix.
        """

        if name is None:
            base = plugin or func or "step"
            name = f"{base}_{len(self.steps)}"
        step: dict = {
            "name": name,
            "params": params or {},
        }
        if func is not None:
            step["func"] = func
        if module is not None:
            step["module"] = module
        if plugin is not None:
            step["plugin"] = plugin
        if depends_on:
            step["depends_on"] = list(depends_on)
        self.steps.append(step)

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

    def freeze_step(self, index: int) -> None:
        """Disable execution of the step at ``index`` without removing it."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        self.steps[index]["frozen"] = True

    def defrost_step(self, index: int) -> None:
        """Re-enable execution of a previously frozen step."""
        if index < 0 or index >= len(self.steps):
            raise IndexError("index out of range")
        self.steps[index]["frozen"] = False

    # Dependency resolution -------------------------------------------------

    def _build_dependency_graph(
        self, steps: list[dict]
    ) -> tuple[nx.DiGraph, dict[str, dict]]:
        """Return a directed graph representing step dependencies.

        Parameters
        ----------
        steps:
            Sequence of step dictionaries. Each must include a ``name`` field and
            may list dependency names in ``depends_on``.

        Returns
        -------
        graph, name_to_step:
            ``networkx.DiGraph`` with edges from dependencies to dependents and a
            mapping from step name to the original step dictionary.
        """

        graph = nx.DiGraph()
        name_to_step: dict[str, dict] = {}
        for idx, step in enumerate(steps):
            name = step.get("name") or f"step_{idx}"
            if name in name_to_step:
                raise ValueError(f"Duplicate step name '{name}'")
            step["name"] = name
            graph.add_node(name)
            name_to_step[name] = step
        for step in steps:
            for dep in step.get("depends_on", []):
                if dep not in name_to_step:
                    raise ValueError(
                        f"Step '{step['name']}' depends on unknown step '{dep}'"
                    )
                graph.add_edge(dep, step["name"])
        return graph, name_to_step

    def _topological_sort(self, steps: list[dict]) -> list[dict]:
        graph, name_to_step = self._build_dependency_graph(steps)
        try:
            order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            cycle = nx.find_cycle(graph)
            chain = " -> ".join(n for n, _ in cycle)
            chain += f" -> {cycle[0][0]}"
            raise ValueError(f"Dependency cycle detected: {chain}")
        return [name_to_step[name] for name in order]

    def execute(
        self,
        marble: Any | None = None,
        *,
        metrics_visualizer: "MetricsVisualizer | None" = None,
        benchmark_iterations: int | None = None,
        preallocate_neurons: int = 0,
        preallocate_synapses: int = 0,
        log_callback: Callable[[str], None] | None = None,
        debug_hook: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        results: list[Any] = []
        self._summaries = []
        self._benchmarks = []
        core = None
        if log_callback is not None:
            log_callback(f"GPU available: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if marble is not None and hasattr(marble, "get_core"):
            core = marble.get_core()
            if preallocate_neurons:
                core.neuron_pool.preallocate(preallocate_neurons)
            if preallocate_synapses:
                core.synapse_pool.preallocate(preallocate_synapses)
        # Resolve dependencies before execution
        self.steps = self._topological_sort(self.steps)
        for idx, step in enumerate(self.steps):
            if step.get("frozen"):
                continue
            wait_for_prefetch()
            module_name = step.get("module")
            func_name = step.get("func")
            params = step.get("params", {})
            start = time.perf_counter()
            if "plugin" in step:
                plugin_name = step["plugin"]
                plugin_cls = pipeline_plugins.get_plugin(plugin_name)
                plugin: pipeline_plugins.PipelinePlugin = plugin_cls(**params)
                plugin.initialise(device=device, marble=marble)
                result = plugin.execute(device=device, marble=marble)
                plugin.teardown()
                func_name = plugin_name
            else:
                if func_name is None:
                    raise ValueError("Step missing 'func' or 'plugin'")
                result = self._execute_function(module_name, func_name, marble, params)
            if hasattr(result, "next_batch") and hasattr(result, "is_finished"):

                async def _drain(step):
                    batches = []
                    async for batch in step:
                        batches.append(batch)
                    step.close()
                    return batches

                result = asyncio.run(_drain(result))
            runtime = time.perf_counter() - start
            results.append(result)
            if log_callback is not None:
                log_callback(f"Step {idx}: {func_name} finished in {runtime:.3f}s")
            if debug_hook is not None:
                debug_hook(idx, result)
            if metrics_visualizer is not None:
                metrics_visualizer.update(
                    {"pipeline_step": idx, "step_runtime": runtime}
                )
            if global_workspace.workspace is not None:
                global_workspace.workspace.publish(
                    "pipeline", {"index": idx, "step": func_name}
                )
            if benchmark_iterations and core is not None:
                _, msg_time = benchmark_message_passing(
                    core, iterations=benchmark_iterations, warmup=1
                )
                self._benchmarks.append(
                    {"step": func_name, "runtime": runtime, "msg_time": msg_time}
                )
            summary: dict | None = None
            if isinstance(result, list) and result and isinstance(result[0], tuple):
                summary = {"step": func_name, "num_pairs": len(result)}
            elif hasattr(result, "summary") and callable(result.summary):
                try:
                    info = result.summary()
                    summary = {"step": func_name, **info}
                except Exception:
                    pass
            if summary:
                self._summaries.append(summary)
        return results

    def _execute_function(
        self, module_name: str | None, func_name: str, marble: Any, params: dict
    ) -> Any:
        module = (
            importlib.import_module(module_name) if module_name else marble_interface
        )
        if not hasattr(module, func_name):
            raise ValueError(f"Unknown function: {func_name}")
        func = getattr(module, func_name)
        sig = inspect.signature(func)
        use_gpu = torch.cuda.is_available()
        kwargs = {}
        if "device" in sig.parameters and "device" not in params:
            kwargs["device"] = "cuda" if use_gpu else "cpu"
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

    def dataset_summaries(self) -> list[dict]:
        """Return summaries for dataset-producing steps from last run."""
        return list(self._summaries)

    def benchmarks(self) -> list[dict]:
        """Return benchmark results collected during :meth:`execute`."""
        return list(self._benchmarks)

    def diff_config(self, other_steps: list[dict]) -> str:
        """Return unified diff between ``other_steps`` and ``self.steps``."""
        a = json.dumps(other_steps, indent=2, sort_keys=True).splitlines(keepends=True)
        b = json.dumps(self.steps, indent=2, sort_keys=True).splitlines(keepends=True)
        return "".join(difflib.unified_diff(a, b, fromfile="before", tofile="after"))
