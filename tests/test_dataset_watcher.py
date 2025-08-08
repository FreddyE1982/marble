import os
from dataset_watcher import DatasetWatcher


def test_changed_files_detection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("a")
    watcher = DatasetWatcher(data)
    assert watcher.has_changed()
    assert set(watcher.changed_files()) == {"a.txt"}
    assert watcher.total_files() == 1

    # No changes
    assert not watcher.has_changed()
    assert watcher.changed_files() == []

    # Modify existing and add new file
    (data / "a.txt").write_text("b")
    (data / "b.txt").write_text("c")
    assert watcher.has_changed()
    assert set(watcher.changed_files()) == {"a.txt", "b.txt"}
    assert watcher.total_files() == 2
