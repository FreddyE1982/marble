from __future__ import annotations

"""Utilities for encrypting and decrypting datasets.

This module provides AES-GCM based encryption helpers that operate on
PyTorch tensors. Tensors are always moved to CPU for encryption and
can be restored to their original device during decryption.
"""

import base64
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class EncryptedTensor:
    """Container holding encrypted tensor data."""

    nonce: bytes
    ciphertext: bytes
    shape: Tuple[int, ...]
    dtype: str
    device: str

    def to_json(self) -> str:
        """Serialise the encrypted tensor to a JSON string."""
        return json.dumps(
            {
                "nonce": base64.b64encode(self.nonce).decode("utf-8"),
                "ciphertext": base64.b64encode(self.ciphertext).decode("utf-8"),
                "shape": self.shape,
                "dtype": self.dtype,
                "device": self.device,
            }
        )

    @staticmethod
    def from_json(data: str) -> "EncryptedTensor":
        """Restore an :class:`EncryptedTensor` from a JSON string."""
        obj = json.loads(data)
        return EncryptedTensor(
            nonce=base64.b64decode(obj["nonce"]),
            ciphertext=base64.b64decode(obj["ciphertext"]),
            shape=tuple(obj["shape"]),
            dtype=obj["dtype"],
            device=obj["device"],
        )


def generate_key() -> str:
    """Generate a new base64 encoded AES-256-GCM key."""
    key = AESGCM.generate_key(bit_length=256)
    return base64.urlsafe_b64encode(key).decode("utf-8")


def load_key_from_env(env_var: str = "DATASET_ENCRYPTION_KEY") -> bytes:
    """Load a base64 encoded key from an environment variable."""
    key_b64 = os.environ.get(env_var)
    if not key_b64:
        raise KeyError(f"Environment variable {env_var} not set")
    return base64.urlsafe_b64decode(key_b64)


def encrypt_tensor(tensor: torch.Tensor, key: bytes) -> EncryptedTensor:
    """Encrypt a tensor using AES-256-GCM.

    Parameters
    ----------
    tensor:
        Tensor to encrypt. It may reside on CPU or GPU. The tensor is moved to
        CPU for encryption and reconstructed on the original device during
        decryption.
    key:
        32-byte key for AES-256-GCM.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    cpu_tensor = tensor.detach().to("cpu")
    buffer = cpu_tensor.numpy().tobytes()
    ciphertext = aesgcm.encrypt(nonce, buffer, None)
    return EncryptedTensor(
        nonce=nonce,
        ciphertext=ciphertext,
        shape=tuple(cpu_tensor.shape),
        dtype=str(cpu_tensor.dtype).replace("torch.", ""),
        device=str(tensor.device),
    )


def decrypt_tensor(enc: EncryptedTensor, key: bytes, device: Optional[torch.device] = None) -> torch.Tensor:
    """Decrypt an :class:`EncryptedTensor` back into a tensor."""
    aesgcm = AESGCM(key)
    buffer = aesgcm.decrypt(enc.nonce, enc.ciphertext, None)
    dtype = getattr(torch, enc.dtype)
    tensor = torch.frombuffer(buffer, dtype=dtype).clone().reshape(enc.shape)
    target_device = torch.device(device) if device is not None else torch.device(enc.device)
    return tensor.to(target_device)
