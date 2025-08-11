import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_loader import create_marble_from_config


def test_federated_learning_enabled_adds_trainer():
    marble = create_marble_from_config(
        overrides={"federated_learning": {"enabled": True, "rounds": 1, "local_epochs": 1}}
    )
    assert hasattr(marble, "federated_trainer")
    assert marble.federated_trainer is not None
