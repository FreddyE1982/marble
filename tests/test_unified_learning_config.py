import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_loader import create_marble_from_config
from learning_plugins import LearningModule, register_learning_module


class DummyLearner(LearningModule):
    def train_step(self, *args, device, marble=None):  # pragma: no cover - simple stub
        return 0.0


def test_unified_learning_from_config(tmp_path):
    register_learning_module("dummy", DummyLearner)
    log_file = tmp_path / "ul.jsonl"
    overrides = {
        "unified_learning": {
            "enabled": True,
            "gating_hidden": 5,
            "log_path": str(log_file),
            "learners": {"d": "dummy"},
        }
    }
    marble = create_marble_from_config(overrides=overrides)
    assert hasattr(marble, "unified_learner")
    ul = marble.unified_learner
    assert ul.gate[0].out_features == 5
    assert ul.log_path == str(log_file)
