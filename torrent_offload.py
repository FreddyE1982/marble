import random
from typing import Dict, List, Set

class BrainTorrentClient:
    """Simple client that holds assigned brain parts."""

    def __init__(self, client_id: str, tracker: 'BrainTorrentTracker'):
        self.client_id = client_id
        self.tracker = tracker
        self.parts: Set[int] = set()
        self.online = False

    def connect(self):
        """Register with the tracker."""
        if not self.online:
            self.online = True
            self.tracker.register_client(self)

    def disconnect(self):
        """Unregister from the tracker and clear parts."""
        if self.online:
            self.tracker.deregister_client(self.client_id)
            self.parts.clear()
            self.online = False

    def add_part(self, part: int):
        self.parts.add(part)

    def remove_part(self, part: int):
        self.parts.discard(part)


class BrainTorrentTracker:
    """Tracker that distributes brain parts among connected clients."""

    def __init__(self):
        self.clients: Dict[str, BrainTorrentClient] = {}
        self.part_to_client: Dict[int, str] = {}

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

    def _assign_part(self, part: int) -> None:
        if not self.clients:
            return
        client_id = random.choice(list(self.clients.keys()))
        self.part_to_client[part] = client_id
        self.clients[client_id].add_part(part)

    def _redistribute_part(self, part: int) -> None:
        self.part_to_client[part] = None
        self._assign_part(part)

    # ---------------------------------------------------------------------
    def get_client_parts(self, client_id: str) -> List[int]:
        return [p for p, cid in self.part_to_client.items() if cid == client_id]

