import torch

from bit_tensor_streaming_dataset import BitTensorStreamingDataset


class MockHFStreamingDataset:
    """Minimal mock of a HuggingFace streaming dataset."""

    def __init__(self, data):
        self._data = list(data)

    def __iter__(self):
        return iter(self._data)

    def select(self, indices):
        return (self._data[i] for i in indices if 0 <= i < len(self._data))


def make_dataset(n=5):
    return MockHFStreamingDataset([(i, i + 1) for i in range(n)])


def test_seek_operations():
    ds = BitTensorStreamingDataset(make_dataset(), batch_size=1)
    ds.seek_to(2)
    inp, tgt = next(iter(ds))
    assert ds.encoder.decode_tensor(inp) == 2
    assert ds.encoder.decode_tensor(tgt) == 3
    assert isinstance(inp, torch.Tensor)

    ds.seek_backward(1)
    inp, tgt = next(iter(ds))
    assert ds.encoder.decode_tensor(inp) == 1
    assert isinstance(inp, torch.Tensor)

    ds.seek_forward(2)
    inp, tgt = next(iter(ds))
    assert ds.encoder.decode_tensor(inp) == 3
    assert isinstance(inp, torch.Tensor)


def test_virtual_batches():
    ds = BitTensorStreamingDataset(make_dataset(), batch_size=1, virtual_batch_size=2)
    batch = ds.get_virtual_batch(1, stream=False)
    assert len(batch) == 2
    first = ds.encoder.decode_tensor(batch[0][0])
    second = ds.encoder.decode_tensor(batch[1][0])
    assert (first, second) == (2, 3)
    assert isinstance(batch[0][0], torch.Tensor)

    streamed = list(ds.get_virtual_batch(0, stream=True))
    assert ds.encoder.decode_tensor(streamed[0][0]) == 0
    assert isinstance(streamed[0][0], torch.Tensor)

    # Re-access previously requested batch
    again = ds.get_virtual_batch(1, stream=False)
    assert ds.encoder.decode_tensor(again[0][0]) == 2
