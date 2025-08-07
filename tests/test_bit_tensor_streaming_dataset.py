import torch

from bit_tensor_streaming_dataset import BitTensorStreamingDataset


def pair_stream(n=3):
    for i in range(n):
        yield {"input": i}, {"target": i + 1}


def test_streaming_single_records():
    ds = BitTensorStreamingDataset(pair_stream(3), batch_size=1)
    items = list(ds)
    assert len(items) == 3
    for inp, tgt in items:
        assert inp.shape[1] == 8
        assert tgt.shape[1] == 8
        assert isinstance(inp, torch.Tensor)


def test_streaming_batches():
    ds = BitTensorStreamingDataset(pair_stream(5), batch_size=2)
    batches = list(ds)
    assert len(batches) == 3
    first_inputs, first_targets = batches[0]
    assert first_inputs.shape[0] == 2
    last_inputs, _ = batches[-1]
    assert last_inputs.shape[0] == 1
