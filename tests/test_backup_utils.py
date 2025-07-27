import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backup_utils import BackupScheduler
from marble_base import MetricsVisualizer


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


def test_metrics_visualizer_backup(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    mv = MetricsVisualizer(log_dir=str(log_dir), backup_dir=str(tmp_path / "bk"), backup_interval=0.1)
    mv.update({"loss": 1.0})
    time.sleep(0.25)
    mv.close()
    backups = list((tmp_path / "bk").glob("backup_*"))
    assert backups
