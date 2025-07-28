import hmac
from typing import Union


def constant_time_compare(val1: Union[str, bytes], val2: Union[str, bytes]) -> bool:
    """Return ``True`` if ``val1`` and ``val2`` are equal using constant-time comparison."""
    if isinstance(val1, str):
        val1 = val1.encode()
    if isinstance(val2, str):
        val2 = val2.encode()
    return hmac.compare_digest(val1, val2)
