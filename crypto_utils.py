import hmac
from typing import Union


def constant_time_compare(val1: Union[str, bytes], val2: Union[str, bytes]) -> bool:
    """Return ``True`` if ``val1`` and ``val2`` are equal using constant-time comparison."""
    if isinstance(val1, str):
        val1 = val1.encode()
    if isinstance(val2, str):
        val2 = val2.encode()
    return hmac.compare_digest(val1, val2)


def _expand_key(key: bytes, length: int) -> bytes:
    """Return ``key`` repeated or truncated to ``length`` bytes."""
    if len(key) == 0:
        raise ValueError("key must not be empty")
    repeats = length // len(key)
    remainder = length % len(key)
    return key * repeats + key[:remainder]


def encrypt_bytes(data: bytes, key: Union[str, bytes]) -> bytes:
    """Return ``data`` XORed with ``key`` using constant time operations."""
    if isinstance(key, str):
        key = key.encode()
    expanded = _expand_key(key, len(data))
    return constant_time_xor(data, expanded)


def decrypt_bytes(data: bytes, key: Union[str, bytes]) -> bytes:
    """Inverse of :func:`encrypt_bytes`."""
    return encrypt_bytes(data, key)


def constant_time_xor(data1: Union[bytes, bytearray], data2: Union[bytes, bytearray]) -> bytes:
    """Return the XOR of two byte sequences using a constant-time loop."""
    if not isinstance(data1, (bytes, bytearray)) or not isinstance(data2, (bytes, bytearray)):
        raise TypeError("Inputs must be bytes or bytearray")
    if len(data1) != len(data2):
        raise ValueError("Inputs must have the same length")

    result = bytearray(len(data1))
    for i in range(len(data1)):
        result[i] = data1[i] ^ data2[i]
    return bytes(result)
