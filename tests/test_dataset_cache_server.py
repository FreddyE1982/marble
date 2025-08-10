import socket
import time

import requests

from dataset_cache_server import DatasetCacheServer
from dataset_encryption import encrypt_bytes


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_dataset_cache_server_decrypts(tmp_path):
    key = "secret"
    data = b"hello"
    enc = b"ENC" + encrypt_bytes(data, key)
    cache_dir = tmp_path
    path = cache_dir / "test.bin"
    path.write_bytes(enc)

    port = _find_free_port()
    server = DatasetCacheServer(cache_dir=str(cache_dir), encryption_key=key)
    server.start(port=port)
    # Give the server a moment to start
    time.sleep(0.1)
    resp = requests.get(f"http://127.0.0.1:{port}/test.bin")
    assert resp.content == data
    server.stop()
