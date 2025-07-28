"""Utilities for synchronising configuration files across nodes."""

from __future__ import annotations

import shutil
from pathlib import Path


def sync_config(src: str, dest_paths: list[str]) -> None:
    """Copy ``src`` config file to all ``dest_paths``."""
    source = Path(src)
    if not source.is_file():
        raise FileNotFoundError(src)
    for dest in dest_paths:
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_path)

from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class _ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, src: Path, dest_paths: list[Path]) -> None:
        self.src = src
        self.dest_paths = dest_paths

    def on_modified(self, event):
        if Path(event.src_path) == self.src:
            for dest in self.dest_paths:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.src, dest)


class ConfigSyncService:
    """Watch a config file and synchronise changes to other nodes."""

    def __init__(self, src: str, dest_paths: list[str]):
        self.src = Path(src)
        self.dest_paths = [Path(p) for p in dest_paths]
        self._observer = Observer()
        self._handler = _ConfigChangeHandler(self.src, self.dest_paths)

    def start(self) -> None:
        self._observer.schedule(self._handler, str(self.src.parent), recursive=False)
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()

    @classmethod
    def run_in_thread(cls, src: str, dest_paths: list[str]) -> Thread:
        svc = cls(src, dest_paths)
        thread = Thread(target=lambda: (svc.start(), svc._observer.join()))
        thread.daemon = True
        thread.start()
        return thread
