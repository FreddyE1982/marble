import random
from typing import Dict, List, Set
from marble_neuronenblitz import Neuronenblitz
from marble_core import Core
from queue import Queue, Empty, Full
from concurrent.futures import Future
import threading

class BrainTorrentClient:
    """Simple client that holds assigned brain parts and executes them.

    Added asynchronous processing with a buffering queue for tasks.
    """

    def __init__(self, client_id: str, tracker: 'BrainTorrentTracker', buffer_size: int = 10, heartbeat_interval: int = 30):
        self.client_id = client_id
        self.tracker = tracker
        self.parts: Set[int] = set()
        self.neuronenblitzes: Dict[int, Neuronenblitz] = {}
        self.online = False
        self.heartbeat_interval = heartbeat_interval

        # asynchronous processing state
        self.buffer: Queue = Queue(maxsize=buffer_size)
        self.worker_thread = None
        self.running = False

    def connect(self):
        """Register with the tracker."""
        if not self.online:
            self.online = True
            self.tracker.register_client(self)
            self._start_worker()

    def disconnect(self):
        """Unregister from the tracker and clear parts."""
        if self.online:
            self.tracker.deregister_client(self.client_id)
            self.parts.clear()
            self.neuronenblitzes.clear()
            self._stop_worker()
            self.online = False

    def add_part(self, part: int):
        self.parts.add(part)
        core = self.tracker.part_data.get(part)
        if core is not None:
            self.neuronenblitzes[part] = Neuronenblitz(core)

    def remove_part(self, part: int):
        self.parts.discard(part)
        self.neuronenblitzes.pop(part, None)

    def offload(self, core: Core) -> int:
        """Offload a subcore through the tracker."""
        return self.tracker.offload_subcore(core)

    def process(self, value: float, part: int) -> float:
        """Process a value using the assigned sub-brain."""
        nb = self.neuronenblitzes.get(part)
        if nb is None:
            return value
        output, _ = nb.dynamic_wander(value)
        return output

    # -- asynchronous processing -------------------------------------------
    def _start_worker(self) -> None:
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _stop_worker(self) -> None:
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
            self.worker_thread = None

    def _worker_loop(self) -> None:
        while self.running:
            try:
                value, part, future = self.buffer.get(timeout=0.1)
            except Empty:
                continue
            try:
                result = self.process(value, part)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.buffer.task_done()

    def process_async(self, value: float, part: int, timeout: float = 1.0) -> Future:
        """Queue a value for asynchronous processing.

        Returns a Future that will hold the result. Raises BufferError if the
        internal buffer is full.
        """
        fut: Future = Future()
        try:
            self.buffer.put((value, part, fut), timeout=timeout)
        except Full:
            raise BufferError(f"Buffer full for client {self.client_id}")
        return fut


class BrainTorrentTracker:
    """Tracker that distributes brain parts among connected clients."""

    def __init__(self):
        self.clients: Dict[str, BrainTorrentClient] = {}
        self.part_to_client: Dict[int, str] = {}
        self.part_data: Dict[int, Core] = {}
        self.next_part: int = 0

    # -- client management -------------------------------------------------
    def register_client(self, client: BrainTorrentClient) -> None:
        """Add a client to the tracker."""
        self.clients[client.client_id] = client
        # assign any unassigned parts
        for part, cid in list(self.part_to_client.items()):
            if cid is None:
                self._assign_part(part)

    def deregister_client(self, client_id: str) -> None:
        """Remove a client and redistribute its parts."""
        if client_id in self.clients:
            del self.clients[client_id]
        for part, cid in list(self.part_to_client.items()):
            if cid == client_id:
                self._redistribute_part(part)

    # -- part management ---------------------------------------------------
    def add_part(self, part: int) -> None:
        """Add a new brain part to be distributed."""
        if part in self.part_to_client:
            return
        self.part_to_client[part] = None
        self._assign_part(part)

    def offload_subcore(self, core: Core) -> int:
        """Register a new subcore and distribute it."""
        part = self.next_part
        self.next_part += 1
        self.part_data[part] = core
        self.add_part(part)
        return part

    def _assign_part(self, part: int) -> None:
        if not self.clients:
            return
        client_id = random.choice(list(self.clients.keys()))
        self.part_to_client[part] = client_id
        self.clients[client_id].add_part(part)

    def _redistribute_part(self, part: int) -> None:
        old_client = self.part_to_client.get(part)
        if old_client and old_client in self.clients:
            self.clients[old_client].remove_part(part)
        self.part_to_client[part] = None
        self._assign_part(part)

    # ---------------------------------------------------------------------
    def get_client_parts(self, client_id: str) -> List[int]:
        return [p for p, cid in self.part_to_client.items() if cid == client_id]

