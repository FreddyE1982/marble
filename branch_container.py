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
        self, steps: List[dict], device: torch.device, marble: Any, kwargs: dict
    ) -> Any:
        steps_with_device: List[dict] = []
        for s in steps:
            s = dict(s)
            params = dict(s.get("params", {}))
            # allow steps to know their device without overriding explicit params
            params.setdefault("device", device.type)
            s["params"] = params
            steps_with_device.append(s)

        def _execute():
            from pipeline import Pipeline

            pipe = Pipeline(steps_with_device)
            results = pipe.execute(marble=marble, **kwargs)
            return results[-1] if results else None

        return await asyncio.to_thread(_execute)

    async def run(self, marble: Any | None, **kwargs) -> List[Any]:
        devices = self._allocate_devices()
        tasks = [
            self._run_branch(steps, dev, marble, kwargs)
            for steps, dev in zip(self.branches, devices)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                raise r
        return results
