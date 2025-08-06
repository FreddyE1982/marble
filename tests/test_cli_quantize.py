import sys
import types

import cli


class DummyBrain:
    def train(self, *args, **kwargs):
        pass


class DummyMarble:
    def get_brain(self):
        return DummyBrain()


def test_quantize_flag(monkeypatch):
    called = {}

    def fake_create(path, overrides=None):
        called["bits"] = overrides["core"].get("quantization_bits")
        return DummyMarble()

    monkeypatch.setattr(cli, "create_marble_from_config", fake_create)
    monkeypatch.setattr(cli, "load_dataset", lambda path: [])
    monkeypatch.setattr(sys, "argv", ["cli.py", "--quantize", "4"])
    cli.main()
    assert called["bits"] == 4
