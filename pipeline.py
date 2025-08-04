from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import networkx as nx
import torch

import global_workspace
import marble_interface
import pipeline_plugins
from branch_container import BranchContainer
from dataset_loader import wait_for_prefetch
from marble_base import MetricsVisualizer
from marble_core import benchmark_message_passing
from marble_neuronenblitz import Neuronenblitz


class PreStepHook(Protocol):
    """Callable executed before a pipeline step.

    Parameters
    ----------
    step:
        Mutable step specification dictionary.
    marble:
        Optional MARBLE instance supplied to :meth:`Pipeline.execute`.
    device:
        Torch device on which the step will run.
    """

    def __call__(
        self, step: dict, marble: Any | None, device: torch.device
    ) -> None: ...


class PostStepHook(Protocol):
    """Callable executed after a pipeline step.

    Parameters are identical to :class:`PreStepHook` with an additional
    ``result`` argument representing the output of the step.  The hook may
    return a replacement result which will be passed to subsequent hooks and
    ultimately returned to the caller.
    """

    def __call__(
        self, step: dict, result: Any, marble: Any | None, device: torch.device
    ) -> Any: ...


class InteractiveDebugger:
    """Hook-based helper capturing step inputs and outputs.

    The debugger registers as both a pre and post hook on pipeline steps. It
    records the parameters supplied to each step and a summary of the produced
    result including tensor device information. When ``interactive`` is ``True``
    the debugger drops into :mod:`pdb` before and after every step allowing live
    inspection of the execution state.  Setting ``interactive=False`` records
    information without pausing which is useful for automated tests.
    """

    def __init__(self, interactive: bool = True) -> None:
        self.interactive = interactive
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}

    def _summarize(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return {
                "type": "tensor",
                "shape": list(obj.shape),
                "device": obj.device.type,
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, dict):
            return {k: self._summarize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._summarize(v) for v in obj]
        return obj

    def pre_hook(self, step: dict, marble: Any | None, device: torch.device) -> None:
        name = step.get("name") or step.get("func") or step.get("plugin") or "step"
        params = step.get("params", {})
        self.inputs[name] = {"params": self._summarize(params), "device": device.type}
        if self.interactive:
            import pdb

            pdb.set_trace()

    def post_hook(
        self, step: dict, result: Any, marble: Any | None, device: torch.device
    ) -> Any:
        name = step.get("name") or step.get("func") or step.get("plugin") or "step"
        self.outputs[name] = self._summarize(result)
        if self.interactive:
            import pdb

            pdb.set_trace()
        return result


class Pipeline:
    """Sequence of function calls executable with an optional MARBLE instance."""

    def __init__(self, steps: list[dict] | None = None) -> None:
        self.steps: list[dict] = steps or []
        self._summaries: list[dict] = []
        self._benchmarks: list[dict] = []
        # Registered hooks keyed by step name. Order of insertion is preserved
        # to guarantee deterministic execution.
        self._pre_hooks: dict[str, list[PreStepHook]] = {}
        self._post_hooks: dict[str, list[PostStepHook]] = {}

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

    def add_branch(
        self,
        branches: list[list[dict]],
        *,
        merge: dict | None = None,
        name: str | None = None,
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a branching node to the pipeline.

        ``branches`` is a list of sub-pipelines, each itself a list of step
        dictionaries.  ``merge`` optionally describes a merge function or
        plugin invoked after all branches finish.  The merge specification
        mirrors :meth:`add_step` with ``func``/``module`` or ``plugin`` and an
        optional ``params`` mapping passed to the merge callable.
        """

        if name is None:
            name = f"branch_{len(self.steps)}"
        step: dict = {"name": name, "branches": branches}
        if merge is not None:
            step["merge"] = merge
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

    # Hook management -----------------------------------------------------

    def register_pre_hook(self, step_name: str, hook: PreStepHook) -> None:
        """Register ``hook`` to run before ``step_name`` executes."""
        self._pre_hooks.setdefault(step_name, []).append(hook)

    def register_post_hook(self, step_name: str, hook: PostStepHook) -> None:
        """Register ``hook`` to run after ``step_name`` completes."""
        self._post_hooks.setdefault(step_name, []).append(hook)

    def remove_pre_hook(self, step_name: str, hook: PreStepHook) -> None:
        """Remove a previously registered pre-hook."""
        hooks = self._pre_hooks.get(step_name)
        if hooks and hook in hooks:
            hooks.remove(hook)

    def remove_post_hook(self, step_name: str, hook: PostStepHook) -> None:
        """Remove a previously registered post-hook."""
        hooks = self._post_hooks.get(step_name)
        if hooks and hook in hooks:
            hooks.remove(hook)

    def enable_interactive_debugging(
        self, *, interactive: bool = True
    ) -> InteractiveDebugger:
        """Register hooks capturing inputs and outputs for all steps.

        Parameters
        ----------
        interactive:
            When ``True`` the debugger invokes :func:`pdb.set_trace` before and
            after each step allowing manual inspection.  ``False`` records data
            without pausing.

        Returns
        -------
        InteractiveDebugger
            The debugger instance storing captured inputs and outputs.
        """

        debugger = InteractiveDebugger(interactive=interactive)
        for i, step in enumerate(self.steps):
            name = step.get("name") or step.get("func") or step.get("plugin")
            if name is None:
                name = f"step_{i}"
                step["name"] = name
            self.register_pre_hook(name, debugger.pre_hook)
            self.register_post_hook(name, debugger.post_hook)
        return debugger

    # Dataset detection ---------------------------------------------------

    def _dataset_step_indices(self) -> list[int]:
        """Return indices of steps that produce datasets.

        A step is considered a dataset producer if its ``func`` or ``plugin``
        name contains the word ``"dataset"``.  This heuristic keeps the
        pipeline flexible while allowing automatic training loop insertion for
        common dataset loaders and streaming dataset plugins.
        """

        indices: list[int] = []
        for i, step in enumerate(self.steps):
            name = (step.get("func") or step.get("plugin") or "").lower()
            if "dataset" in name:
                indices.append(i)
        return indices

    def _train_neuronenblitz(
        self, nb: Neuronenblitz, dataset, device: torch.device, epochs: int = 1
    ) -> None:
        """Train ``nb`` on ``dataset`` ensuring tensors reside on ``device``."""

        prepared = []
        for item in dataset:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            inp, tgt = item[0], item[1]
            if isinstance(inp, torch.Tensor):
                inp = inp.to(device)
            if isinstance(tgt, torch.Tensor):
                tgt = tgt.to(device)
            prepared.append((inp, tgt))
        if prepared:
            nb.train(prepared, epochs=epochs)

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
        cache_dir: str | Path | None = None,
        export_path: str | Path | None = None,
        export_format: str = "json",
    ) -> list[Any]:
        results: list[Any] = []
        self._summaries = []
        self._benchmarks = []
        core = None
        if log_callback is not None:
            log_callback(f"GPU available: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cache_path: Path | None = Path(cache_dir) if cache_dir is not None else None
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
        if marble is not None and hasattr(marble, "get_core"):
            core = marble.get_core()
            if preallocate_neurons:
                core.neuron_pool.preallocate(preallocate_neurons)
            if preallocate_synapses:
                core.synapse_pool.preallocate(preallocate_synapses)
        # Resolve dependencies before execution
        self.steps = self._topological_sort(self.steps)
        dataset_indices = self._dataset_step_indices()
        total_steps = sum(1 for s in self.steps if not s.get("frozen"))
        exec_idx = 0
        from event_bus import PROGRESS_EVENT, ProgressEvent, global_event_bus

        for idx, step in enumerate(self.steps):
            if step.get("frozen"):
                continue
            step_name = (
                step.get("name")
                or step.get("func")
                or step.get("plugin")
                or f"step_{idx}"
            )
            label = step_name
            global_event_bus.publish(
                PROGRESS_EVENT,
                ProgressEvent(
                    step=label,
                    index=exec_idx,
                    total=total_steps,
                    device=device.type,
                    status="started",
                ).as_dict(),
            )
            wait_for_prefetch()
            for hook in self._pre_hooks.get(step_name, []):
                hook(step, marble, device)
            start = time.perf_counter()
            result = None
            executed = True
            cache_file = None
            if cache_path:
                spec_bytes = json.dumps(step, sort_keys=True).encode("utf-8")
                digest = hashlib.sha256(spec_bytes).hexdigest()
                cache_file = cache_path / f"{idx}_{step_name}_{digest}.pt"
                if cache_file.exists():
                    if log_callback is not None:
                        log_callback(f"Loading cached result for {step_name}")
                    result = torch.load(cache_file, map_location=device)
                    executed = False
            if executed:
                if "branches" in step:
                    container = BranchContainer(step["branches"])
                    branch_outputs = asyncio.run(
                        container.run(
                            marble,
                            metrics_visualizer=metrics_visualizer,
                            benchmark_iterations=benchmark_iterations,
                            log_callback=log_callback,
                            debug_hook=debug_hook,
                        )
                    )
                    merge_spec = step.get("merge")
                    if merge_spec:
                        merge_params = dict(merge_spec.get("params", {}))
                        merge_params["branches"] = branch_outputs
                        if "plugin" in merge_spec:
                            plugin_name = merge_spec["plugin"]
                            plugin_cls = pipeline_plugins.get_plugin(plugin_name)
                            plugin: pipeline_plugins.PipelinePlugin = plugin_cls(
                                **merge_params
                            )
                            plugin.initialise(device=device, marble=marble)
                            result = plugin.execute(device=device, marble=marble)
                            plugin.teardown()
                            func_name = plugin_name
                        else:
                            func_name = merge_spec.get("func")
                            module_name = merge_spec.get("module")
                            if func_name is None:
                                raise ValueError(
                                    "Merge step missing 'func' or 'plugin'"
                                )
                            result = self._execute_function(
                                module_name, func_name, marble, merge_params
                            )
                    else:
                        func_name = "branch"
                        result = branch_outputs
                else:
                    module_name = step.get("module")
                    func_name = step.get("func")
                    params = step.get("params", {})
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
                        result = self._execute_function(
                            module_name, func_name, marble, params
                        )
                if cache_file:
                    torch.save(result, cache_file)
            if hasattr(result, "next_batch") and hasattr(result, "is_finished"):

                async def _drain(step):
                    batches = []
                    async for batch in step:
                        batches.append(batch)
                    step.close()
                    return batches

                result = asyncio.run(_drain(result))
            if (
                isinstance(marble, Neuronenblitz)
                and idx in dataset_indices
                and isinstance(result, list)
            ):
                epochs = int(step.get("params", {}).get("epochs", 1))
                self._train_neuronenblitz(marble, result, device, epochs=epochs)
            for hook in self._post_hooks.get(step_name, []):
                result = hook(step, result, marble, device)
            runtime = time.perf_counter() - start if executed else 0.0
            results.append(result)
            global_event_bus.publish(
                PROGRESS_EVENT,
                ProgressEvent(
                    step=label,
                    index=exec_idx,
                    total=total_steps,
                    device=device.type,
                    status="completed",
                ).as_dict(),
            )
            exec_idx += 1
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
        if export_path is not None and marble is not None:
            plugin_cls = pipeline_plugins.get_plugin("export_model")
            exporter = plugin_cls(path=str(export_path), fmt=export_format)
            exporter.initialise(device=device, marble=marble)
            export_result = exporter.execute(device=device, marble=marble)
            exporter.teardown()
            results.append(export_result)
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
