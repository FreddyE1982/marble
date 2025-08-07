from __future__ import annotations

import asyncio
import difflib
import hashlib
import importlib
import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

import networkx as nx
import psutil
import torch

import global_workspace
import marble_interface
import pipeline_plugins
from bit_tensor_dataset import bytes_to_object, tensors_to_bytes
from branch_container import BranchContainer
from dataset_loader import wait_for_prefetch
from marble_base import MetricsVisualizer
from marble_core import TIER_REGISTRY, benchmark_message_passing
from marble_neuronenblitz import Neuronenblitz
from memory_manager import MemoryManager
from pipeline_schema import validate_step_schema
from run_profiler import RunProfiler
from cross_validation import cross_validate


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
        validate_step_schema(step)
        self.steps.append(step)

    def add_macro(
        self,
        name: str,
        steps: list[dict],
        *,
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a macro step composed of multiple sub-steps.

        Parameters
        ----------
        name:
            Unique name identifying the macro step.
        steps:
            List of step dictionaries executed sequentially when the macro
            runs. Each sub-step is validated against the pipeline schema.
        depends_on:
            Optional list of step names that must complete before this macro
            executes.
        """

        for s in steps:
            validate_step_schema(s)
        step: dict = {"name": name, "macro": steps}
        if depends_on:
            step["depends_on"] = list(depends_on)
        validate_step_schema(step)
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
        for i, branch in enumerate(branches):
            for j, s in enumerate(branch):
                if "name" not in s:
                    base = s.get("plugin") or s.get("func") or "step"
                    s["name"] = f"{base}_{i}_{j}"
                validate_step_schema(s)
        validate_step_schema(step)
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

    # ------------------------------------------------------------------
    # Cross-validation convenience

    def run_cross_validation(
        self,
        train_fn: Callable[[Iterable, torch.device], Any],
        metric_fn: Callable[[Any, Iterable, torch.device], float],
        dataset: Sequence,
        *,
        folds: int | None = None,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> List[float]:
        """Execute ``train_fn``/``metric_fn`` across cross-validation folds.

        This helper delegates to :func:`cross_validation.cross_validate` but is
        provided as a method on :class:`Pipeline` to simplify reuse inside
        pipeline-driven experiments. Both functions must respect the supplied
        :class:`torch.device` to maintain CPU/GPU parity.
        """

        return cross_validate(
            train_fn,
            metric_fn,
            dataset,
            folds=folds,
            seed=seed,
            device=device,
        )

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

        def _decode(t: torch.Tensor):
            if isinstance(t, torch.Tensor) and t.dtype in {
                torch.uint8,
                torch.int32,
                torch.int64,
            }:
                try:
                    return bytes_to_object(tensors_to_bytes(t.cpu()))
                except Exception:
                    return t
            return t

        # Support datasets streamed as batches of tensors.
        if (
            isinstance(dataset, list)
            and dataset
            and isinstance(dataset[0], dict)
            and "inputs" in dataset[0]
        ):
            for batch in dataset:
                inputs = batch.get("inputs")
                targets = batch.get("targets")
                if not isinstance(inputs, torch.Tensor) or not isinstance(
                    targets, torch.Tensor
                ):
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

    # ------------------------------------------------------------------
    # Resource estimation
    def estimate_resources(
        self,
        marble: Any | None = None,
        device: torch.device | None = None,
        memory_manager: "MemoryManager | None" = None,
    ) -> int:
        """Estimate memory requirements for all steps.

        Steps may define a ``<func>_estimate`` helper or, for plugins, an
        ``estimate_memory`` method. Macros and branches are evaluated
        recursively. When ``memory_manager`` is provided its
        ``notify_allocation`` method is called for each estimate.
        """

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ordered = self._topological_sort(self.steps)
        total = 0
        for step in ordered:
            if step.get("frozen"):
                continue
            if "macro" in step or "branches" in step:
                total += self._estimate_step(step, marble, device, memory_manager)
            else:
                est = self._estimate_step(step, marble, device, None)
                total += est
                if memory_manager is not None and est:
                    memory_manager.notify_allocation(est)
        return total

    def _estimate_step(
        self,
        step: dict,
        marble: Any | None,
        device: torch.device,
        memory_manager: "MemoryManager | None",
    ) -> int:
        if "macro" in step:
            sub = Pipeline(step["macro"])
            return sub.estimate_resources(marble, device, memory_manager)
        if "branches" in step:
            total = 0
            for branch in step["branches"]:
                sub = Pipeline(branch)
                total += sub.estimate_resources(marble, device, memory_manager)
            return total
        params = step.get("params", {})
        if "plugin" in step:
            plugin_cls = pipeline_plugins.get_plugin(step["plugin"])
            plugin: pipeline_plugins.PipelinePlugin = plugin_cls(**params)
            plugin.initialise(device=device, marble=marble)
            try:
                if hasattr(plugin, "estimate_memory"):
                    return int(plugin.estimate_memory(device=device, marble=marble))
            finally:
                plugin.teardown()
            return 0
        func_name = step.get("func")
        module_name = step.get("module")
        if not func_name:
            return 0
        est_name = f"{func_name}_estimate"
        module = (
            importlib.import_module(module_name) if module_name else marble_interface
        )
        if not hasattr(module, est_name):
            return 0
        est_func = getattr(module, est_name)
        sig = inspect.signature(est_func)
        kwargs = {}
        if "device" in sig.parameters and "device" not in params:
            kwargs["device"] = device.type
        for name, p in sig.parameters.items():
            if name == "marble" and marble is not None:
                kwargs[name] = marble
                continue
            if name in params:
                kwargs[name] = params[name]
            elif p.default is inspect.Parameter.empty:
                raise ValueError(f"Missing parameter: {name}")
        return int(est_func(**kwargs))

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
        max_gpu_concurrency: int | None = None,
        memory_manager: "MemoryManager | None" = None,
        run_profile_path: str | Path | None = None,
        run_profiler: "RunProfiler | None" = None,
        pre_estimate: bool = True,
        default_memory_limit_mb: float | None = None,
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
        profiler = run_profiler or (
            RunProfiler(run_profile_path) if run_profile_path else None
        )
        if marble is not None and hasattr(marble, "get_core"):
            core = marble.get_core()
            if preallocate_neurons:
                core.neuron_pool.preallocate(preallocate_neurons)
            if preallocate_synapses:
                core.synapse_pool.preallocate(preallocate_synapses)
        # Resolve dependencies before execution
        self.steps = self._topological_sort(self.steps)
        if pre_estimate and memory_manager is not None:
            self.estimate_resources(marble, device, memory_manager)
        dataset_indices = self._dataset_step_indices()
        total_steps = sum(1 for s in self.steps if not s.get("frozen"))
        exec_idx = 0
        from event_bus import PROGRESS_EVENT, ProgressEvent, global_event_bus

        for idx, step in enumerate(self.steps):
            validate_step_schema(step)
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
            if profiler:
                profiler.start(step_name, device)
            for hook in self._pre_hooks.get(step_name, []):
                hook(step, marble, device)
            process = psutil.Process()
            cpu_before = process.memory_info().rss
            gpu_before = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
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
                if step.get("isolated"):
                    # Execute the step in a separate process for fault tolerance
                    result = self._run_isolated_step(step, marble, device)
                    func_name = (
                        step.get("func")
                        or step.get("plugin")
                        or step.get("name")
                        or "step"
                    )
                else:
                    tier_name = step.get("tier")
                    if tier_name and tier_name in TIER_REGISTRY:
                        tier = TIER_REGISTRY[tier_name]
                        tier.connect()
                        try:
                            result = tier.run_step(step, marble, device)
                        finally:
                            tier.close()
                        func_name = tier_name
                    elif "macro" in step:
                        sub_cache = cache_path / step_name if cache_path else None
                        sub_pipeline = Pipeline(step["macro"])
                        # Share hooks so macros participate in global pre/post
                        sub_pipeline._pre_hooks = self._pre_hooks
                        sub_pipeline._post_hooks = self._post_hooks
                        result = sub_pipeline.execute(
                            marble,
                            metrics_visualizer=metrics_visualizer,
                            benchmark_iterations=benchmark_iterations,
                            preallocate_neurons=0,
                            preallocate_synapses=0,
                            log_callback=log_callback,
                            debug_hook=debug_hook,
                            cache_dir=sub_cache,
                            max_gpu_concurrency=max_gpu_concurrency,
                            memory_manager=memory_manager,
                            run_profiler=profiler,
                            pre_estimate=False,
                        )
                        self._summaries.extend(sub_pipeline._summaries)
                        self._benchmarks.extend(sub_pipeline._benchmarks)
                        func_name = "macro"
                    elif "branches" in step:
                        container = BranchContainer(step["branches"])
                        branch_outputs = asyncio.run(
                            container.run(
                                marble,
                                metrics_visualizer=metrics_visualizer,
                                benchmark_iterations=benchmark_iterations,
                                log_callback=log_callback,
                                debug_hook=debug_hook,
                                max_gpu_concurrency=max_gpu_concurrency,
                                memory_manager=memory_manager,
                                run_profiler=profiler,
                                pre_estimate=False,
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
                            plugin: pipeline_plugins.PipelinePlugin = plugin_cls(
                                **params
                            )
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
            cpu_after = process.memory_info().rss
            gpu_after = (
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            )
            usage = max(cpu_after - cpu_before, gpu_after - gpu_before)
            limit = step.get("memory_limit_mb", default_memory_limit_mb)
            if memory_manager is not None:
                memory_manager.notify_step_usage(step_name, usage, limit)
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
            if summary is None:
                summary = {"step": func_name}
            summary["memory_mb"] = usage / (1024**2)
            self._summaries.append(summary)
            if profiler:
                profiler.end()
        if export_path is not None and marble is not None:
            plugin_cls = pipeline_plugins.get_plugin("export_model")
            exporter = plugin_cls(path=str(export_path), fmt=export_format)
            exporter.initialise(device=device, marble=marble)
            export_result = exporter.execute(device=device, marble=marble)
            exporter.teardown()
            results.append(export_result)
        if profiler:
            profiler.save()
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

    def hyperparameter_search(
        self,
        param_grid: Mapping[str, Iterable[Any]],
        score_func: Callable[[list[Any]], float],
        *,
        search: str = "grid",
        num_samples: int | None = None,
        marble: Any | None = None,
        **execute_kwargs,
    ) -> list[tuple[dict[str, Any], float]]:
        """Run a hyperparameter search over pipeline step parameters.

        ``param_grid`` maps keys of the form ``"step.param"`` to iterables of
        possible values.  For each sampled combination the pipeline is executed
        and ``score_func`` is called with the resulting list of step outputs.
        The function should return a numeric score where lower values are
        considered better.

        Parameters
        ----------
        param_grid:
            Mapping from ``"step.param"`` keys to value options.
        score_func:
            Callable evaluating pipeline results and returning a numeric score.
        search:
            Either ``"grid"`` or ``"random"`` to select the search strategy.
        num_samples:
            Number of random samples when ``search="random"``.
        marble:
            Optional MARBLE instance forwarded to :meth:`execute`.
        execute_kwargs:
            Additional keyword arguments forwarded to :meth:`execute`.
        """

        from copy import deepcopy

        from hyperparameter_search import grid_search, random_search

        def _to_cpu(obj: Any) -> Any:
            """Recursively move tensors in ``obj`` to CPU."""
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_cpu(v) for v in obj]
            return obj

        def run(params: dict[str, Any]) -> float:
            original = deepcopy(self.steps)
            try:
                for key, value in params.items():
                    step_name, param_name = key.split(".", 1)
                    for s in self.steps:
                        name = s.get("name") or s.get("func") or s.get("plugin") or ""
                        if name == step_name:
                            s.setdefault("params", {})[param_name] = value
                            break
                    else:
                        raise KeyError(f"Unknown step '{step_name}'")
                results = self.execute(marble, **execute_kwargs)
                score = float(score_func(_to_cpu(results)))
            finally:
                self.steps = deepcopy(original)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            return score

        if search == "grid":
            return grid_search(param_grid, run)
        if search == "random":
            if num_samples is None:
                raise ValueError("num_samples required for random search")
            return random_search(param_grid, run, num_samples)
        raise ValueError("search must be 'grid' or 'random'")

    def rollback(
        self,
        step_name: str,
        cache_dir: str | Path,
        *,
        device: torch.device | None = None,
    ) -> Any:
        """Load cached output of ``step_name`` and discard later results.

        This utility is useful when experimentation produces undesirable
        outcomes. By rolling back to a previous step the pipeline can be
        re-executed from that point without recomputing earlier steps.  Cached
        files for steps after ``step_name`` are deleted so subsequent runs start
        fresh.  The loaded result is returned for immediate inspection.
        """

        cache_path = Path(cache_dir)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ordered = self._topological_sort(self.steps)
        target_file = None
        for idx, step in enumerate(ordered):
            name = (
                step.get("name")
                or step.get("func")
                or step.get("plugin")
                or f"step_{idx}"
            )
            spec_bytes = json.dumps(step, sort_keys=True).encode("utf-8")
            digest = hashlib.sha256(spec_bytes).hexdigest()
            file = cache_path / f"{idx}_{name}_{digest}.pt"
            if name == step_name:
                target_file = file
                remove_from = idx + 1
                break
        if target_file is None or not target_file.exists():
            raise ValueError(f"No cached result for step '{step_name}'")
        # Delete cached outputs of later steps
        for j in range(remove_from, len(ordered)):
            s = ordered[j]
            n = s.get("name") or s.get("func") or s.get("plugin") or f"step_{j}"
            # Remove any cached file regardless of previous digest
            for cached in cache_path.glob(f"{j}_{n}_*.pt"):
                cached.unlink()
            subdir = cache_path / n
            if subdir.exists() and subdir.is_dir():
                shutil.rmtree(subdir)
        return torch.load(target_file, map_location=device)

    def diff_config(self, other_steps: list[dict]) -> str:
        """Return unified diff between ``other_steps`` and ``self.steps``."""
        a = json.dumps(other_steps, indent=2, sort_keys=True).splitlines(keepends=True)
        b = json.dumps(self.steps, indent=2, sort_keys=True).splitlines(keepends=True)
        return "".join(difflib.unified_diff(a, b, fromfile="before", tofile="after"))

    def _run_isolated_step(
        self, step: dict, marble: Any | None, device: torch.device
    ) -> Any:
        """Execute ``step`` in a separate process and return its result.

        The child process constructs a temporary :class:`Pipeline` with the given
        step so that hooks, macros and branches behave identically to normal
        execution.  Running steps in isolated processes protects the main
        pipeline from crashes or resource leaks in individual steps.
        """

        import multiprocessing as mp

        ctx = mp.get_context("spawn" if device.type == "cuda" else "fork")
        q: mp.Queue = ctx.Queue()
        step_copy = dict(step)
        step_copy.pop("isolated", None)

        def _target(q, step, marble, device, pre_hooks, post_hooks):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            p = Pipeline([step])
            p._pre_hooks = pre_hooks
            p._post_hooks = post_hooks
            try:
                result = p.execute(marble)[0]
                q.put(("ok", result, p._summaries, p._benchmarks))
            except Exception as exc:  # pragma: no cover - propagated
                q.put(("err", exc))

        proc = ctx.Process(
            target=_target,
            args=(q, step_copy, marble, device, self._pre_hooks, self._post_hooks),
        )
        proc.start()
        proc.join()
        status, *payload = q.get()
        if status == "err":
            raise payload[0]
        result, summaries, benchmarks = payload
        self._summaries.extend(summaries)
        self._benchmarks.extend(benchmarks)
        return result
