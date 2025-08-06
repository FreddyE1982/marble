from dataset_sync_service import detect_dataset_changes, sync_remote_dataset
from dataset_loader import export_dataset, load_dataset


def test_sync_remote_dataset(monkeypatch, tmp_path):
    local = tmp_path / "local.csv"
    remote = tmp_path / "remote.csv"
    export_dataset([(1, 2), (3, 4)], str(local))
    export_dataset([(1, 2)], str(remote))

    progress = {"total": 0, "count": 0}

    def fake_tqdm(iterable, total=None, **kwargs):
        progress["total"] = total
        for item in iterable:
            progress["count"] += 1
            yield item

    monkeypatch.setattr("dataset_sync_service.tqdm", fake_tqdm)

    ops = sync_remote_dataset(str(local), str(remote))
    assert ops == 1
    assert progress["total"] == 1
    assert progress["count"] == 1
    assert load_dataset(str(remote)) == load_dataset(str(local))


def test_detect_dataset_changes(tmp_path):
    local = tmp_path / "local.csv"
    remote = tmp_path / "remote.csv"
    export_dataset([(1, 2), (3, 4)], str(local))
    export_dataset([(1, 2)], str(remote))
    ops = detect_dataset_changes(str(local), str(remote))
    assert any(op["op"] == "add" for op in ops)
