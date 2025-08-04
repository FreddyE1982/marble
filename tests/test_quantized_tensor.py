import torch

from quantized_tensor import QuantizedTensor


def _devices():
    return [torch.device("cpu")] + (
        [torch.device("cuda")] if torch.cuda.is_available() else []
    )


def test_round_trip():
    for device in _devices():
        original = torch.tensor(
            [-1.0, -0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32
        )
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


def test_cpu_gpu_consistency():
    """CPU and GPU quantization should produce equivalent results."""
    if not torch.cuda.is_available():
        return
    cpu_tensor = torch.randn(32, dtype=torch.float32, device="cpu")
    gpu_tensor = cpu_tensor.to("cuda")
    cpu_qt = QuantizedTensor.from_tensor(cpu_tensor, bit_width=4)
    gpu_qt = QuantizedTensor.from_tensor(gpu_tensor, bit_width=4)
    cpu_restored = cpu_qt.to_dense()
    gpu_restored = gpu_qt.to_dense().to("cpu")
    atol = max(cpu_qt.scale, gpu_qt.scale)
    assert torch.allclose(cpu_restored, gpu_restored, atol=atol)


def test_gpu_to_cpu_serialization_roundtrip():
    """Quantized tensor serialized on GPU loads correctly on CPU."""
    if not torch.cuda.is_available():
        return
    gpu_tensor = torch.randn(16, dtype=torch.float32, device="cuda")
    qt_gpu = QuantizedTensor.from_tensor(gpu_tensor, bit_width=6)
    state = qt_gpu.state_dict()
    # simulate loading on a CPU-only machine
    state["device"] = "cpu"
    loaded = QuantizedTensor.from_state_dict(state)
    restored = loaded.to_dense()
    assert restored.device.type == "cpu"
    atol = qt_gpu.scale
    assert torch.allclose(gpu_tensor.to("cpu"), restored, atol=atol)


def test_to_device_roundtrip():
    """QuantizedTensor.to should move data across devices without loss."""
    if not torch.cuda.is_available():
        return
    tensor = torch.randn(20, dtype=torch.float32, device="cuda")
    qt_gpu = QuantizedTensor.from_tensor(tensor, bit_width=5)
    qt_cpu = qt_gpu.to("cpu")
    assert qt_cpu.device.type == "cpu"
    qt_back = qt_cpu.to("cuda")
    assert qt_back.device.type == "cuda"
    restored = qt_back.to_dense()
    atol = qt_back.scale
    assert torch.allclose(tensor, restored, atol=atol)
