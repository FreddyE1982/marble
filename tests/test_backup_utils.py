import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backup_utils import BackupScheduler


def test_backup_scheduler_creates_backup(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "file.txt").write_text("data")
    sched = BackupScheduler(str(src), str(dst), interval_sec=0.1)
    sched.start()
    time.sleep(0.25)
    sched.stop()
    backups = list(dst.glob("backup_*/file.txt"))
    assert backups and backups[0].read_text() == "data"
