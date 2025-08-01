import torch
from bit_tensor_dataset import BitTensorDataset


def test_ann_search(tmp_path):
    ds = BitTensorDataset([(i.to_bytes(1,'little'), i.to_bytes(1,'little')) for i in range(10)])
    ds.build_ann_index()
    target = ds.encode_object( b"\x05")
    idxs = ds.nearest_neighbors(target, k=3)
    assert 5 in idxs
