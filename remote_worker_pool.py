from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Iterable, List

import cloudpickle


class _PreprocessWorker(mp.Process):
    """Worker process executing preprocessing jobs.

    The worker receives pickled callables and their single argument through a
    ``multiprocessing.Pipe``. Results or raised exceptions are sent back to the
    parent process. Workers run as daemons so they exit automatically if the
    parent dies.
    """

    def __init__(self, conn: mp.connection.Connection) -> None:
        super().__init__(daemon=True)
        self._conn = conn

    def run(self) -> None:  # pragma: no cover - run in child process
        while True:
            try:
                msg = self._conn.recv()
            except EOFError:
                break
            if msg[0] == "shutdown":
                break
            job_id, func_bytes, item = msg
            func = cloudpickle.loads(func_bytes)
            try:
                result = func(item)
                self._conn.send(("ok", job_id, result))
            except Exception as exc:  # pragma: no cover - worker reports errors
                self._conn.send(("err", job_id, exc))


class RemoteWorkerPool:
    """Execute preprocessing functions across isolated worker processes.

    The pool uses simple RPC over ``multiprocessing.Pipe`` connections. Workers
    are restarted transparently when they terminate unexpectedly which allows
    long running preprocessing jobs to recover from crashes.
    """

    def __init__(self, num_workers: int, *, max_retries: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.num_workers = num_workers
        self.max_retries = max_retries
        self._workers: List[_PreprocessWorker] = []
        self._conns: List[mp.connection.Connection] = []
        self._spawn_all()

    def _spawn_all(self) -> None:
        for _ in range(self.num_workers):
            parent, child = mp.Pipe()
            worker = _PreprocessWorker(child)
            worker.start()
            self._workers.append(worker)
            self._conns.append(parent)

    def _restart_worker(self, idx: int) -> None:
        try:
            self._workers[idx].kill()
            self._workers[idx].join(timeout=1)
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        parent, child = mp.Pipe()
        worker = _PreprocessWorker(child)
        worker.start()
        self._workers[idx] = worker
        self._conns[idx] = parent

    def map(self, func: Callable[[Any], Any], items: Iterable[Any]) -> List[Any]:
        func_bytes = cloudpickle.dumps(func)
        items_list = list(items)
        results: List[Any] = [None] * len(items_list)
        pending = set(range(len(items_list)))
        active: dict[int, int | None] = {w: None for w in range(self.num_workers)}
        for idx, item in enumerate(items_list):
            widx = idx % self.num_workers
            self._conns[widx].send((idx, func_bytes, item))
            active[widx] = idx
        retries: dict[int, int] = {i: self.max_retries for i in pending}
        while pending:
            for widx, conn in enumerate(self._conns):
                if not pending:
                    break
                if conn.poll(0.1):
                    try:
                        status, job_id, payload = conn.recv()
                    except (EOFError, OSError):
                        job_id = active[widx]
                        self._restart_worker(widx)
                        conn = self._conns[widx]
                        if job_id is not None:
                            conn.send((job_id, func_bytes, items_list[job_id]))
                            active[widx] = job_id
                        continue
                    if status == "ok":
                        results[job_id] = payload
                        pending.remove(job_id)
                        active[widx] = None
                    else:  # err
                        retries[job_id] -= 1
                        if retries[job_id] < 0:
                            raise payload
                        conn.send((job_id, func_bytes, items_list[job_id]))
                        active[widx] = job_id
            for widx, worker in enumerate(self._workers):
                if not worker.is_alive():
                    job_id = active[widx]
                    self._restart_worker(widx)
                    if job_id is not None:
                        self._conns[widx].send((job_id, func_bytes, items_list[job_id]))
                        active[widx] = job_id
        return results

    def shutdown(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("shutdown", None, None))
            except Exception:  # pragma: no cover - best effort
                pass
        for worker in self._workers:
            worker.join(timeout=1)
