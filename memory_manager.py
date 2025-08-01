class MemoryManager:
    """Simple memory manager tracking upcoming allocations."""

    def __init__(self):
        self.reservations = []

    def notify_allocation(self, size: int) -> None:
        """Record a future allocation in bytes."""
        self.reservations.append(size)

    def total_reserved(self) -> int:
        return sum(self.reservations)
