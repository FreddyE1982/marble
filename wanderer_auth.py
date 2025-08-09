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


class SessionManager:
    """Minimal in-memory session tracker.

    Sessions are identified by ``wanderer_id`` and represented by a signed token.
    The manager keeps track of expiration times and can renew tokens on demand.

    Parameters
    ----------
    secret:
        Shared secret used for HMAC signing.
    session_timeout:
        Number of seconds after which an inactive session expires.
    """

    def __init__(self, secret: str, session_timeout: float = 300.0) -> None:
        self.secret = secret
        self.session_timeout = session_timeout
        # Mapping of wanderer_id -> (token, expiration timestamp)
        self._sessions: dict[str, tuple[str, float]] = {}

    def start(self, wanderer_id: str) -> str:
        """Create a new session token for ``wanderer_id``."""

        token = generate_token(self.secret, wanderer_id)
        self._sessions[wanderer_id] = (
            token,
            time.time() + self.session_timeout,
        )
        return token

    def verify(self, token: str) -> bool:
        """Validate ``token`` and refresh its expiration."""

        if not verify_token(self.secret, token, self.session_timeout):
            return False
        _, wanderer_id, _ = token.split(":", 2)
        stored = self._sessions.get(wanderer_id)
        if stored is None or stored[0] != token:
            return False
        self._sessions[wanderer_id] = (
            token,
            time.time() + self.session_timeout,
        )
        return True

    def renew(self, token: str) -> str | None:
        """Return a new token if the existing ``token`` is still valid."""

        if not self.verify(token):
            return None
        _, wanderer_id, _ = token.split(":", 2)
        return self.start(wanderer_id)

    def cleanup(self) -> None:
        """Remove expired sessions from the internal store."""

        now = time.time()
        expired = [wid for wid, (_, exp) in self._sessions.items() if exp < now]
        for wid in expired:
            del self._sessions[wid]

    def active_sessions(self) -> dict[str, float]:
        """Return a mapping of active wanderer IDs to expiration timestamps."""

        return {wid: exp for wid, (_, exp) in self._sessions.items()}

    def revoke(self, token: str) -> bool:
        """Invalidate the session identified by ``token``."""

        for wid, (stored_token, _) in list(self._sessions.items()):
            if stored_token == token:
                del self._sessions[wid]
                return True
        return False
