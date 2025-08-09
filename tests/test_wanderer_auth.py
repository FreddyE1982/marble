import time

from wanderer_auth import SessionManager, generate_token, verify_token


def test_generate_and_verify_token():
    secret = "abc"
    token = generate_token(secret, "w1", timestamp=0)
    assert verify_token(secret, token, max_age=10)


def test_token_expiry():
    secret = "abc"
    token = generate_token(secret, "w1", timestamp=time.time() - 100)
    assert verify_token(secret, token, max_age=200)
    assert not verify_token(secret, token, max_age=50)


def test_token_tamper_detection():
    secret = "abc"
    token = generate_token(secret, "w1", timestamp=0)
    parts = token.split(":")
    parts[2] = "deadbeef"
    bad_token = ":".join(parts)
    assert not verify_token(secret, bad_token)


def test_session_lifecycle():
    mgr = SessionManager("abc", session_timeout=0.1)
    token = mgr.start("w1")
    assert mgr.verify(token)
    # Renewal should extend the session and return a new token
    new_token = mgr.renew(token)
    assert new_token is not None
    assert mgr.verify(new_token)
    # Expire the session
    time.sleep(0.2)
    assert not mgr.verify(new_token)


def test_active_and_revoke_session():
    mgr = SessionManager("abc", session_timeout=1.0)
    token = mgr.start("w1")
    assert "w1" in mgr.active_sessions()
    assert mgr.revoke(token)
    assert "w1" not in mgr.active_sessions()
    # Revoking again should fail
    assert not mgr.revoke(token)
