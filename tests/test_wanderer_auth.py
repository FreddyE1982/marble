import time

from wanderer_auth import generate_token, verify_token


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
