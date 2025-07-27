import time
import types
from remote_offload import RemoteBrainClient


def test_remote_client_latency_tracking(monkeypatch):
    def fake_post(url, json=None, timeout=0):
        time.sleep(0.01)
        class Res:
            def json(self):
                return {"output": 1.0}
        return Res()

    monkeypatch.setattr("requests.post", fake_post)
    client = RemoteBrainClient("http://localhost", track_latency=True, compression_enabled=False)
    client.process(0.2)
    client.process(0.3)
    assert len(client.latencies) == 2
    assert all(lat >= 0.01 for lat in client.latencies)
    assert client.average_latency >= 0.01
