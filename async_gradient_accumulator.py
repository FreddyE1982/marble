from __future__ import annotations

import asyncio
from typing import Callable

import torch


class AsyncGradientAccumulator:
    """Asynchronously accumulate gradients before applying optimiser steps.

    Each call to :meth:`add_batch` schedules loss computation and backward
    propagation in a background thread using :func:`asyncio.to_thread`.  The
    optimiser step is executed once ``accumulation_steps`` batches have been
    processed.  Inputs and targets are automatically moved to the selected
    device.  When CUDA is available, transfers use non-blocking copies to keep
    the pipeline scheduler responsive.

    Parameters
    ----------
    model:
        PyTorch module whose parameters receive gradients.
    optimizer:
        Optimiser updating ``model`` parameters.
    loss_fn:
        Callable computing the loss given model outputs and targets.
    accumulation_steps:
        Number of batches to accumulate before calling ``optimizer.step``.
    device:
        Target device.  ``None`` selects ``"cuda"`` when available otherwise
        ``"cpu"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        accumulation_steps: int = 1,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.device = (
            torch.device("cuda")
            if device is None and torch.cuda.is_available()
            else torch.device(device or "cpu")
        )
        self.model.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self._counter = 0
        self._lock = asyncio.Lock()

    async def add_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Accumulate gradients for ``inputs``/``targets``.

        Returns the detached loss value for monitoring.
        """

        async with self._lock:
            if inputs.device != self.device:
                if self.device.type == "cuda" and inputs.device.type == "cpu":
                    inputs = inputs.pin_memory().to(self.device, non_blocking=True)
                    targets = targets.pin_memory().to(self.device, non_blocking=True)
                else:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

            def _backward() -> float:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                return float(loss.item())

            loss_val = await asyncio.to_thread(_backward)
            self._counter += 1
            if self._counter >= self.accumulation_steps:
                await asyncio.to_thread(self.optimizer.step)
                self.optimizer.zero_grad(set_to_none=True)
                self._counter = 0
            return loss_val

    async def flush(self) -> None:
        """Apply remaining gradients and reset accumulation."""

        async with self._lock:
            if self._counter > 0:
                await asyncio.to_thread(self.optimizer.step)
                self.optimizer.zero_grad(set_to_none=True)
                self._counter = 0
