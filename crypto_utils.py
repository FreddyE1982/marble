import hmac
from typing import Union


def constant_time_compare(val1: Union[str, bytes], val2: Union[str, bytes]) -> bool:
    """Return ``True`` if ``val1`` and ``val2`` are equal using constant-time comparison."""
    if isinstance(val1, str):
        val1 = val1.encode()
    if isinstance(val2, str):
        val2 = val2.encode()
    return hmac.compare_digest(val1, val2)


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
