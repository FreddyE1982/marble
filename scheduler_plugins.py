from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Type


class SchedulerPlugin:
    """Abstract interface for asynchronous task schedulers.

    Implementations dispatch callables or coroutines to background execution
    while respecting CPU or GPU placement of the underlying operations. The
    ``schedule`` method returns a :class:`concurrent.futures.Future` that can be
    awaited for the result on both CPU and CUDA devices.
    """

    def start(self) -> None:
        """Prepare the scheduler for accepting work."""

    def schedule(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        """Dispatch ``fn`` for asynchronous execution."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Stop the scheduler and release associated resources."""


_SCHEDULER_REGISTRY: Dict[str, Type[SchedulerPlugin]] = {}


def register_scheduler(name: str, cls: Type[SchedulerPlugin]) -> None:
    """Register ``cls`` under ``name``."""

    _SCHEDULER_REGISTRY[name] = cls


def get_scheduler_cls(name: str) -> Type[SchedulerPlugin]:
    """Return the scheduler class identified by ``name``."""

    return _SCHEDULER_REGISTRY[name]


class ThreadSchedulerPlugin(SchedulerPlugin):
    """Scheduler backed by :class:`ThreadPoolExecutor`."""

    def __init__(self, max_workers: int | None = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def start(self) -> None:  # pragma: no cover - nothing to initialise
        pass

    def schedule(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


class AsyncIOSchedulerPlugin(SchedulerPlugin):
    """Scheduler running an asyncio event loop in a background thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def schedule(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        if asyncio.iscoroutinefunction(fn):
            coro = fn(*args, **kwargs)
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        return asyncio.run_coroutine_threadsafe(
            self.loop.run_in_executor(None, fn, *args, **kwargs), self.loop
        )

    def shutdown(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


register_scheduler("thread", ThreadSchedulerPlugin)
register_scheduler("asyncio", AsyncIOSchedulerPlugin)

_current: SchedulerPlugin | None = None


def configure_scheduler(name: str) -> SchedulerPlugin:
    """Initialise the global scheduler plugin by ``name``."""

    global _current
    if _current is not None:
        _current.shutdown()
    cls = get_scheduler_cls(name)
    _current = cls()
    _current.start()
    return _current


def get_scheduler() -> SchedulerPlugin:
    """Return the active global scheduler, initialising the default if needed."""

    global _current
    if _current is None:
        _current = ThreadSchedulerPlugin()
        _current.start()
    return _current
