import base64
import time

import pytest
import torch

from dataset_encryption import decrypt_tensor, encrypt_tensor, generate_key


def _benchmark_roundtrip(tensor: torch.Tensor, key: bytes):
    start = time.perf_counter()
    _ = tensor.detach().to("cpu").numpy().tobytes()
    baseline = time.perf_counter() - start

    start = time.perf_counter()
    enc = encrypt_tensor(tensor, key)
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = decrypt_tensor(enc, key, device=tensor.device)
    dec_time = time.perf_counter() - start

    return baseline, enc_time, dec_time


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encryption_overhead(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tensor = torch.randn(1024, 1024, device=device)
    key = base64.urlsafe_b64decode(generate_key())
    baseline, enc_time, dec_time = _benchmark_roundtrip(tensor, key)
    assert enc_time <= baseline * 50
    assert dec_time <= baseline * 50
