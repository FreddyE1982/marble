import torch

import model_refresh
from dataset_watcher import DatasetWatcher
from model_refresh import auto_refresh


def test_auto_refresh_strategy(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("0")
    watcher = DatasetWatcher(data)
    watcher.has_changed()  # establish baseline snapshot

    model = torch.nn.Linear(1, 1)
    dataset = [(torch.tensor([0.0]), torch.tensor([0.0]))]

    calls = []

    def fake_full(m, d, **kw):
        calls.append("full")
        return m

    def fake_inc(m, d, **kw):
        calls.append("inc")
        return m

    monkeypatch.setattr(model_refresh, "full_retrain", fake_full)
    monkeypatch.setattr(model_refresh, "incremental_update", fake_inc)

    # Force full retrain
    (data / "a.txt").write_text("1")
    auto_refresh(model, dataset, watcher, strategy="full")
    assert calls == ["full"]

    # Force incremental update
    (data / "a.txt").write_text("2")
    auto_refresh(model, dataset, watcher, strategy="incremental")
    assert calls == ["full", "inc"]

    # Auto strategy triggers full retrain when ratio exceeds threshold
    (data / "a.txt").write_text("3")
    (data / "b.txt").write_text("1")
    auto_refresh(model, dataset, watcher, strategy="auto", change_threshold=0.4)
    assert calls[-1] == "full"


def test_auto_refresh_no_change(monkeypatch, tmp_path):
    """Ensure no refresh occurs when the dataset is unmodified."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("0")
    watcher = DatasetWatcher(data)
    watcher.has_changed()  # establish baseline snapshot

    model = torch.nn.Linear(1, 1)
    dataset = [(torch.tensor([0.0]), torch.tensor([0.0]))]

    calls: list[str] = []

    def fake_full(m, d, **kw):
        calls.append("full")
        return m

    def fake_inc(m, d, **kw):
        calls.append("inc")
        return m

    monkeypatch.setattr(model_refresh, "full_retrain", fake_full)
    monkeypatch.setattr(model_refresh, "incremental_update", fake_inc)

    model, refreshed = auto_refresh(model, dataset, watcher, strategy="auto")
    assert refreshed is False
    assert calls == []
