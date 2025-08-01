from marble import DataLoader


def test_lazy_decode(monkeypatch):
    dl = DataLoader(lazy_decode=True)
    called = {}

    def decode_eager(t):
        called["x"] = True
        return b"data"

    monkeypatch.setattr(dl, "_decode_eager", decode_eager)
    encoded = dl.encode(b"data")
    lazy = dl.decode(encoded)
    assert "x" not in called
    assert lazy.value == b"data"
    assert "x" in called
