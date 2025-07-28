import importlib

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
import self_monitoring


def test_self_monitor_updates_context():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    importlib.reload(self_monitoring)
    self_monitoring.activate(nb, history_size=5)
    nb.update_context(arousal=0.2, stress=0.1, reward=0.3)
    self_monitoring.log_error(0.5)
    ctx = nb.context_history[-1]
    assert "markers" in ctx and ctx["markers"]
    assert ctx["markers"][0]["mean_error"] == ctx["markers"][0]["mean_error"]

