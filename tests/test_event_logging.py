import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_base import MetricsVisualizer
from marble import DataLoader
from tests.test_core_functions import minimal_params


def test_epoch_events_logged():
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    mv = MetricsVisualizer()
    brain = Brain(core, nb, DataLoader())
    brain.metrics_visualizer = mv
    brain.train([(0.1, 0.1)], epochs=1)
    names = [e[0] for e in mv.events]
    assert "epoch_start" in names and "epoch_end" in names
