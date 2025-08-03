import torch
from quantized_tensor import QuantizedTensor


def _devices():
    return [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


def test_round_trip():
    for device in _devices():
        original = torch.linspace(-1.0, 1.0, steps=17, device=device)
        qt = QuantizedTensor.from_tensor(original, bit_width=4)
        restored = qt.to_dense()
        assert torch.allclose(original, restored, atol=qt.scale)


def test_state_dict_serialization():
    tensor = torch.randn(10)
    qt = QuantizedTensor.from_tensor(tensor, bit_width=8)
    state = qt.state_dict()
    loaded = QuantizedTensor.from_state_dict(state)
    assert torch.allclose(qt.to_dense(), loaded.to_dense())
    assert qt.bit_width == loaded.bit_width
    assert qt.shape == loaded.shape


def test_to_bits_length():
    t = torch.arange(0, 16, dtype=torch.float32)
    qt = QuantizedTensor.from_tensor(t, bit_width=4)
    bits = qt.to_bits()
    assert bits.dtype == torch.uint8
    expected_len = (t.numel() * 4 + 7) // 8
    assert bits.numel() == expected_len
