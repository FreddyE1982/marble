import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble import log_metrics


def test_log_metrics_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_metrics(1, 0, 0.1, 12.5)
    log_file = tmp_path / "training_log.txt"
    assert log_file.exists()
    text = log_file.read_text()
    assert "Epoch 1, Tar 0" in text
