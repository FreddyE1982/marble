"""Authentication helpers for remote wanderers.

Provides minimal HMAC-based token generation and verification that
is independent of CPU/GPU availability.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def generate_token(
    secret: str, wanderer_id: str, timestamp: float | None = None
) -> str:
    """Return ``token`` authorising ``wanderer_id``.

    Parameters
    ----------
    secret:
        Shared secret used for HMAC signing.
    wanderer_id:
        Identifier of the remote wanderer.
    timestamp:
        Optional POSIX timestamp.  Defaults to ``time.time()``.
    """

    if timestamp is None:
        timestamp = time.time()
    payload = f"{wanderer_id}:{int(timestamp)}".encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"{int(timestamp)}:{wanderer_id}:{signature}"


def verify_token(secret: str, token: str, max_age: float = 300.0) -> bool:
    """Validate ``token`` and enforce ``max_age`` seconds lifetime."""

    try:
        ts_str, wanderer_id, signature = token.split(":", 2)
        timestamp = int(ts_str)
    except ValueError:
        return False
    expected = hmac.new(
        secret.encode(), f"{wanderer_id}:{timestamp}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False
    return (time.time() - timestamp) <= max_age
