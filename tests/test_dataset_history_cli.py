import torch

from bit_tensor_dataset import BitTensorDataset
from dataset_history_cli import list_history, redo_cmd, revert_cmd, undo_cmd


def test_history_and_revert(tmp_path):
    ds = BitTensorDataset([(1, 2)])
    ds.add_pair(3, 4)
    ids = ds.history_ids()
    assert len(ids) == 2
    path = tmp_path / "ds.pt"
    torch.save(ds, path)
    undo_cmd(str(path), 1, str(path))
    ds_undo = torch.load(path, weights_only=False)
    assert ds_undo.raw_data == [(1, 2)]
    redo_cmd(str(path), 1, str(path))
    ds_redo = torch.load(path, weights_only=False)
    assert ds_redo.raw_data == [(1, 2), (3, 4)]
    revert_cmd(str(path), ids[0], str(path))
    ds_revert = torch.load(path, weights_only=False)
    assert ds_revert.raw_data == [(1, 2)]
    hist = list_history(str(path))
    assert ids[0] in hist and ids[1] in hist
