import requests
import time
import types
import pytest
from remote_offload import RemoteBrainClient


def test_connect_retry_interval(monkeypatch):
    attempts = []
    def fake_get(url, timeout, headers=None, verify=True):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise requests.RequestException
        resp = types.SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None)
        return resp
    sleeps = []
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    client = RemoteBrainClient("http://server", max_retries=3, connect_retry_interval=0.1)
    client.connect()
    assert len(attempts) == 3
    assert sleeps == [0.1, 0.1]


def test_ping_uses_timeout_and_ssl_verify(monkeypatch):
    captured = {}
    def fake_get(url, timeout, headers=None, verify=True):
        captured["timeout"] = timeout
        captured["verify"] = verify
        resp = types.SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None)
        return resp
    monkeypatch.setattr(requests, "get", fake_get)
    client = RemoteBrainClient("http://server", heartbeat_timeout=7, ssl_verify=False)
    client.ping()
    assert captured["timeout"] == 7
    assert captured["verify"] is False
