import os

from config_loader import load_config
from marble_main import MARBLE


def test_metrics_visualizer_disabled_via_env():
    os.environ["MARBLE_DISABLE_METRICS"] = "1"
    cfg = load_config()
    marble = MARBLE(cfg["core"])
    assert marble.get_metrics_visualizer() is None
    del os.environ["MARBLE_DISABLE_METRICS"]

