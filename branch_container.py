from __future__ import annotations

import asyncio
from typing import Any, Iterable, List

import torch


class BranchContainer:
    """Execute multiple sub-pipelines concurrently.

    Each branch is a list of step dictionaries compatible with :class:`Pipeline`.
    The container assigns devices to branches based on available resources and
    gathers their results once all branches have completed.  If a branch raises
    an exception the container cancels remaining branches and re-raises the
    error, ensuring the pipeline halts promptly.
    """

    def __init__(self, branches: Iterable[List[dict]]) -> None:
        self.branches: List[List[dict]] = [list(b) for b in branches]

    # ------------------------------------------------------------------
    # Device planning
    def _free_gpu_memory(self) -> int:
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return total - reserved

    def _allocate_devices(self) -> List[torch.device]:
        num = len(self.branches)
        if torch.cuda.is_available():
            free_mem = self._free_gpu_memory()
            mem_per_branch = free_mem // max(1, num)
            # Require at least 512MB free per branch to use GPU
            if mem_per_branch > 512 * 1024 * 1024:
                return [torch.device("cuda")] * num
        return [torch.device("cpu")] * num

    # ------------------------------------------------------------------
    async def _run_branch(
        self,
        steps: List[dict],
        device: torch.device,
        marble: Any,
        kwargs: dict,
        sem: asyncio.Semaphore | None = None,
        *,
        branch_idx: int,
        num_branches: int,
        pre_estimate: bool,
    ) -> Any:
        steps_with_device: List[dict] = []
        for s in steps:
            s = dict(s)
            params = dict(s.get("params", {}))
            # allow steps to know their device and shard without overriding explicit params
            params.setdefault("device", device.type)
            params.setdefault("num_shards", num_branches)
            params.setdefault("shard_index", branch_idx)
            s["params"] = params
            steps_with_device.append(s)

        async def _async_execute() -> Any:
            def _execute():
                from pipeline import Pipeline

                pipe = Pipeline(steps_with_device)
                results = pipe.execute(marble=marble, pre_estimate=pre_estimate, **kwargs)
                return results[-1] if results else None

            return await asyncio.to_thread(_execute)

        if device.type == "cuda" and sem is not None:
            async with sem:
                return await _async_execute()
        return await _async_execute()

    async def run(
        self,
        marble: Any | None,
        *,
        max_gpu_concurrency: int | None = None,
        pre_estimate: bool = True,
        **kwargs,
    ) -> List[Any]:
        devices = self._allocate_devices()
        sem = (
            asyncio.Semaphore(max_gpu_concurrency)
            if max_gpu_concurrency is not None and any(d.type == "cuda" for d in devices)
            else None
        )
        tasks = [
            self._run_branch(
                steps,
                dev,
                marble,
                kwargs,
                sem,
                branch_idx=idx,
                num_branches=len(self.branches),
                pre_estimate=pre_estimate,
            )
            for idx, (steps, dev) in enumerate(zip(self.branches, devices))
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                raise r
        return results
