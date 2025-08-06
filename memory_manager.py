class MemoryManager:
    """Track and enforce memory usage for pipeline steps."""

    def __init__(self, step_quota_mb: float | None = None) -> None:
        self.reservations: list[int] = []
        self.step_quota_mb = step_quota_mb
        self.step_usage: dict[str, int] = {}

    def notify_allocation(self, size: int) -> None:
        """Record a future allocation in bytes."""
        self.reservations.append(size)

    def notify_step_usage(
        self, step_name: str, size_bytes: int, limit_mb: float | None = None
    ) -> None:
        """Record actual memory consumed by ``step_name``.

        Parameters
        ----------
        step_name:
            Identifier of the executed step.
        size_bytes:
            Number of bytes allocated while executing the step.
        limit_mb:
            Optional override for the memory limit in megabytes. When ``None``
            the manager's ``step_quota_mb`` value is used. Exceeding the limit
            raises :class:`MemoryError`.
        """

        self.step_usage[step_name] = size_bytes
        limit = limit_mb if limit_mb is not None else self.step_quota_mb
        if limit is not None and size_bytes > limit * 1024**2:
            raise MemoryError(
                f"Step '{step_name}' exceeded memory limit of {limit} MB with "
                f"{size_bytes / 1024 ** 2:.2f} MB"
            )

    def total_reserved(self) -> int:
        return sum(self.reservations)
