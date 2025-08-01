import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import DataLoader

class Custom:
    def __init__(self, value: str):
        self.value = value


def _encode(obj: Custom) -> bytes:
    return obj.value.encode("utf-8")


def _decode(data: bytes) -> Custom:
    return Custom(data.decode("utf-8"))


DataLoader.register_plugin(Custom, _encode, _decode)

def test_custom_plugin_roundtrip():
    dl = DataLoader()
    obj = Custom("hello")
    tensor = dl.encode(obj)
    out = dl.decode(tensor)
    assert isinstance(out, Custom)
    assert out.value == "hello"
