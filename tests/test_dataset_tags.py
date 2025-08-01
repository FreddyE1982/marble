from bit_tensor_dataset import BitTensorDataset


def test_dataset_tags():
    ds = BitTensorDataset([(1, 2)], tags=[["a/b"]])
    assert ds.data[0].tags == ["a/b"]
    ds.filter_by_tag("a/b")
    assert len(ds) == 1
    ds.filter_by_tag("other")
    assert len(ds) == 0
