import pytest
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


class CountingHFStreamingDataset(MockHFStreamingDataset):
    """Mock dataset that records which indices were accessed via ``select``."""

    def __init__(self, data):
        super().__init__(data)
        self.selections: list[int] = []

    def select(self, indices):
        def gen():
            for i in indices:
                self.selections.append(i)
                yield self._data[i]

        return gen()


class SkipTakeDataset:
    """Dataset exposing ``skip``/``take`` like HuggingFace ``IterableDataset``."""

    def __init__(self, data):
        self._data = list(data)

    def __iter__(self):
        return iter(self._data)

    def skip(self, n):
        return SkipTakeDataset(self._data[n:])

    def take(self, n):
        return SkipTakeDataset(self._data[:n])


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


@pytest.mark.parametrize("vb", [1, 2, 3])
def test_virtual_batch_sizes(vb):
    ds = BitTensorStreamingDataset(make_dataset(7), virtual_batch_size=vb)
    batch = ds.get_virtual_batch(1, stream=False)
    assert len(batch) == min(vb, 7 - vb)
    start = vb
    for i, (inp, _tgt) in enumerate(batch):
        assert ds.encoder.decode_tensor(inp) == start + i
        assert isinstance(inp, torch.Tensor)


def test_streaming_is_lazy():
    base = CountingHFStreamingDataset([(i, i + 1) for i in range(10)])
    ds = BitTensorStreamingDataset(base, virtual_batch_size=3)
    gen = ds.get_virtual_batch(1, stream=True)
    first = next(gen)
    assert base.selections == [3]
    assert ds.encoder.decode_tensor(first[0]) == 3
    rest = list(gen)
    assert base.selections == [3, 4, 5]
    assert len(rest) == 2


def test_iterates_over_virtual_batches():
    ds = BitTensorStreamingDataset(make_dataset(5), virtual_batch_size=2)
    batches = list(iter(ds))
    assert len(batches) == 3
    second = batches[1]
    decoded = [
        ds.encoder.decode_tensor(second[0][i]) for i in range(second[0].shape[0])
    ]
    assert decoded == [2, 3]


def test_skip_take_dict_records():
    data = [{"input": i, "target": i + 1} for i in range(6)]
    base = SkipTakeDataset(data)
    ds = BitTensorStreamingDataset(base, virtual_batch_size=2)
    batch = ds.get_virtual_batch(1)
    decoded = [ds.encoder.decode_tensor(x[0]) for x in batch]
    assert decoded == [2, 3]
    ds.seek_to(4)
    batch2 = ds.get_virtual_batch(2)
    assert ds.encoder.decode_tensor(batch2[0][0]) == 4
    assert ds.encoder.decode_tensor(batch2[0][1]) == 5
