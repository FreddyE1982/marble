import pytest
import torch

from dataset_watcher import DatasetWatcher
from marble_core import Core
from marble_neuronenblitz.core import Neuronenblitz


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neuronenblitz_refresh_on_dataset_change(tmp_path, device):
    """Neuronenblitz resets when the watched dataset changes."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("0")
    watcher = DatasetWatcher(data)
    watcher.has_changed()  # establish baseline snapshot

    core = Core(params={"width": 1, "height": 1})
    nb = Neuronenblitz(core, auto_update=True, dataset_path=str(data))

    reset_calls = {"count": 0}

    def fake_reset() -> None:
        reset_calls["count"] += 1

    nb.reset_learning_state = fake_reset

    # No change yet
    assert nb.refresh_on_dataset_change(watcher) is False
    assert reset_calls["count"] == 0

    # Modify dataset and expect refresh
    (data / "a.txt").write_text("1")
    assert nb.refresh_on_dataset_change(watcher) is True
    assert reset_calls["count"] == 1
