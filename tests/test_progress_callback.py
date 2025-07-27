import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_train_progress_callback():
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    progress = []
    brain.train([(0.1, 0.2)], epochs=3, progress_callback=lambda p: progress.append(p))
    assert progress and progress[-1] == 1.0
    assert len(progress) == 3
