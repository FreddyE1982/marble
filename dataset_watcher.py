import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Iterable


class DatasetWatcher:
    """Monitor a dataset directory for file changes using checksums.

    The watcher recursively hashes all files in ``path`` and persists a
    mapping of relative file names to their SHA256 digests. Subsequent
    calls to :meth:`has_changed` compare the current mapping to the
    previously stored one and record which files have been added, removed
    or modified. The implementation is entirely CPU based so the behaviour
    is identical on systems with or without GPUs.
    """

    def __init__(self, path: str | os.PathLike[str], checksum_path: Optional[str | os.PathLike[str]] = None) -> None:
        self.path = Path(path)
        if checksum_path is None:
            checksum_path = self.path / ".dataset_state.json"
        self.checksum_path = Path(checksum_path)
        self.checksum_path.parent.mkdir(parents=True, exist_ok=True)
        self._changed: list[str] | None = None
        self._total_files: int | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_snapshot(self) -> Dict[str, str]:
        """Return a mapping of ``relative_path -> sha256`` for all files."""
        snapshot: Dict[str, str] = {}
        if not self.path.exists():
            return snapshot
        for file in sorted(p for p in self.path.rglob("*") if p.is_file()):
            sha = hashlib.sha256()
            with file.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
            snapshot[str(file.relative_to(self.path))] = sha.hexdigest()
        return snapshot

    def _read_snapshot(self) -> Dict[str, str]:
        if not self.checksum_path.exists():
            return {}
        text = self.checksum_path.read_text().strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Backwards compatibility with legacy single checksum files
            return {"__legacy_checksum__": text}

    def _write_snapshot(self, snap: Dict[str, str]) -> None:
        self.checksum_path.write_text(json.dumps(snap, sort_keys=True))

    def _diff(self, old: Dict[str, str], new: Dict[str, str]) -> Iterable[str]:
        paths = set(old) | set(new)
        for p in paths:
            if old.get(p) != new.get(p):
                yield p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def has_changed(self) -> bool:
        """Return ``True`` if the dataset contents differ from the last run."""
        current = self._compute_snapshot()
        previous = self._read_snapshot()
        changed = list(self._diff(previous, current))
        self._changed = changed
        self._total_files = len(current)
        if changed:
            self._write_snapshot(current)
            return True
        return False

    def changed_files(self) -> list[str]:
        """Return list of files that changed since the previous snapshot."""
        if self._changed is None:
            self.has_changed()
        return self._changed or []

    def total_files(self) -> int:
        """Return the total number of tracked files in the dataset."""
        if self._total_files is None:
            self.has_changed()
        return self._total_files or 0
