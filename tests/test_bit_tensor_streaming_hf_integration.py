import datasets

from bit_tensor_streaming_dataset import BitTensorStreamingDataset


def test_hf_iterable_dataset_integration():
    hf_ds = datasets.Dataset.from_dict(
        {"input": [0, 1, 2, 3], "target": [1, 2, 3, 4]}
    ).to_iterable_dataset()
    ds = BitTensorStreamingDataset(hf_ds, virtual_batch_size=2)
    batch = ds.get_virtual_batch(1)
    decoded = [ds.encoder.decode_tensor(x[0]) for x in batch]
    assert decoded == [2, 3]
