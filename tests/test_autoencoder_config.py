import marble_interface
from types import SimpleNamespace


def _dummy_marble():
    return SimpleNamespace(
        get_core=lambda: object(),
        get_neuronenblitz=lambda: object(),
    )


def test_train_autoencoder_uses_config_defaults(monkeypatch):
    cfg = {
        "autoencoder_learning": {
            "enabled": True,
            "epochs": 2,
            "batch_size": 3,
            "noise_std": 0.2,
            "noise_decay": 0.5,
        }
    }
    monkeypatch.setattr(marble_interface, "load_config", lambda: cfg)

    captured = {}

    class DummyLearner:
        def __init__(self, core, nb, noise_std, noise_decay):
            captured["noise_std"] = noise_std
            captured["noise_decay"] = noise_decay
            self.history = [{"loss": 0.0}]

        def train(self, values, epochs=1, batch_size=1):
            captured["epochs"] = epochs
            captured["batch_size"] = batch_size

    monkeypatch.setattr(marble_interface, "AutoencoderLearner", DummyLearner)
    marble_interface.train_autoencoder(_dummy_marble(), [0.1, 0.2, 0.3])
    assert captured == {
        "noise_std": 0.2,
        "noise_decay": 0.5,
        "epochs": 2,
        "batch_size": 3,
    }


def test_train_autoencoder_disabled(monkeypatch):
    cfg = {"autoencoder_learning": {"enabled": False}}
    monkeypatch.setattr(marble_interface, "load_config", lambda: cfg)

    class DummyLearner:
        def __init__(self, *a, **k):  # pragma: no cover - should not be called
            raise AssertionError("should not instantiate when disabled")

    monkeypatch.setattr(marble_interface, "AutoencoderLearner", DummyLearner)
    assert marble_interface.train_autoencoder(_dummy_marble(), [0.1]) == 0.0

