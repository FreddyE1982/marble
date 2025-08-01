"""Utilities for asynchronous data transformations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, List, TypeVar
import torch

_T = TypeVar("_T")
_U = TypeVar("_U")

_executor = ThreadPoolExecutor()


def async_transform(data: Iterable[_T], fn: Callable[[_T], _U]) -> List[_U]:
    """Apply ``fn`` to ``data`` asynchronously using background threads.

    GPU operations are queued on the default CUDA stream when available so that
    transformations run during idle cycles.
    """
    futures = []
    for item in data:
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        futures.append(_executor.submit(fn, item))
    return [f.result() for f in futures]
