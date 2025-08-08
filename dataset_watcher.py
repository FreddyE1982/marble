import hashlib
import os
from pathlib import Path
from typing import Optional


class DatasetWatcher:
    """Monitor a dataset directory for file changes using checksums.

    The watcher recursively hashes all files in ``path``. Whenever
    :meth:`has_changed` is called the current checksum is compared to the
    previously stored one. A new checksum is persisted to ``checksum_path``
    if a change is detected. The implementation is CPU-only and works the
    same regardless of GPU availability.
    """

    def __init__(self, path: str | os.PathLike[str], checksum_path: Optional[str | os.PathLike[str]] = None) -> None:
        self.path = Path(path)
        if checksum_path is None:
            checksum_path = self.path / ".dataset_checksum"
        self.checksum_path = Path(checksum_path)
        self.checksum_path.parent.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self) -> str:
        """Return SHA256 checksum of all files under ``path``."""
        sha = hashlib.sha256()
        if not self.path.exists():
            return sha.hexdigest()
        for file in sorted(p for p in self.path.rglob("*") if p.is_file()):
            sha.update(str(file.relative_to(self.path)).encode())
            with file.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
        return sha.hexdigest()

    def has_changed(self) -> bool:
        """Return ``True`` if the dataset contents differ from the last run."""
        current = self._compute_checksum()
        previous = self.checksum_path.read_text().strip() if self.checksum_path.exists() else None
        if current != previous:
            self.checksum_path.write_text(current)
            return True
        return False
