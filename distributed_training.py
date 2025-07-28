"""Utilities for distributed training."""

from __future__ import annotations

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from marble_core import Core, init_distributed, cleanup_distributed
from marble_neuronenblitz import Neuronenblitz


class DistributedTrainer:
    """Simple wrapper for training ``Neuronenblitz`` models using PyTorch DDP."""

    def __init__(self, params: dict, world_size: int = 1, backend: str = "gloo") -> None:
        self.params = params
        self.world_size = world_size
        self.backend = backend

    def _worker(self, rank: int, data: list[tuple[float, float]]) -> None:
        init_distributed(self.world_size, rank, self.backend)
        core = Core(self.params)
        nb = Neuronenblitz(core)
        for inp, target in data:
            nb.train([(inp, target)], epochs=1)
        state = torch.tensor([s.weight for s in core.synapses], dtype=torch.float32)
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
        state /= self.world_size
        for w, syn in zip(state.tolist(), core.synapses):
            syn.weight = w
        cleanup_distributed()

    def train(self, data: list[tuple[float, float]]) -> None:
        mp.spawn(self._worker, args=(data,), nprocs=self.world_size, join=True)
