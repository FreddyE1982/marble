import base64
import pytest
import torch

from dataset_encryption import (
    encrypt_tensor,
    decrypt_tensor,
    encrypt_bytes,
    decrypt_bytes,
    generate_key,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encrypt_decrypt_tensor_roundtrip(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tensor = torch.randn(4, 5, device=device)
    key = base64.urlsafe_b64decode(generate_key())
    enc = encrypt_tensor(tensor, key)
    dec = decrypt_tensor(enc, key)
    assert dec.device == tensor.device
    assert torch.allclose(dec, tensor)


def test_encrypt_decrypt_bytes_roundtrip():
    key = generate_key()
    payload = b"marble encryption test"
    enc = encrypt_bytes(payload, key)
    dec = decrypt_bytes(enc, key)
    assert dec == payload


def test_decrypt_tensor_with_wrong_key_fails():
    tensor = torch.ones(3)
    key = base64.urlsafe_b64decode(generate_key())
    enc = encrypt_tensor(tensor, key)
    wrong_key = base64.urlsafe_b64decode(generate_key())
    with pytest.raises(Exception):
        decrypt_tensor(enc, wrong_key)


def test_decrypt_bytes_with_wrong_key_fails():
    key = generate_key()
    payload = b"abc"
    enc = encrypt_bytes(payload, key)
    other_key = generate_key()
    with pytest.raises(Exception):
        decrypt_bytes(enc, other_key)
