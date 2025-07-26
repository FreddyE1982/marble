import os
import shutil
import threading
from datetime import datetime

class BackupScheduler:
    """Periodically copy files from ``src_dir`` to ``dst_dir``."""

    def __init__(self, src_dir: str, dst_dir: str, interval_sec: float = 3600.0) -> None:
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.interval_sec = interval_sec
        self.thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self.run_backup()

    def run_backup(self) -> str:
        os.makedirs(self.dst_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(self.dst_dir, f"backup_{timestamp}")
        shutil.copytree(self.src_dir, dst, dirs_exist_ok=True)
        return dst

    def stop(self) -> None:
        if self.thread is not None:
            self._stop_event.set()
            self.thread.join()
            self.thread = None
