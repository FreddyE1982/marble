"""Utilities for asynchronous data transformations."""

from __future__ import annotations

from typing import Callable, Iterable, List, TypeVar

import torch

from scheduler_plugins import get_scheduler

_T = TypeVar("_T")
_U = TypeVar("_U")


def async_transform(data: Iterable[_T], fn: Callable[[_T], _U]) -> List[_U]:
    """Apply ``fn`` to ``data`` asynchronously using the configured scheduler.

    Tasks are dispatched via the active scheduler plugin which may leverage
    threads or an asyncio event loop. GPU operations are queued on the default
    CUDA stream when available so that transformations run during idle cycles on
    both CPU and GPU.
    """
    scheduler = get_scheduler()
    futures = []
    for item in data:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        futures.append(scheduler.schedule(fn, item))
    return [f.result() for f in futures]
