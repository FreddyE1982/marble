import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params
from tqdm import tqdm as std_tqdm


def test_super_evolution_records_metrics():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    import marble_brain as mb
    mb.tqdm = std_tqdm
    brain = Brain(core, nb, DataLoader(), super_evolution_mode=True)
    examples = [(0.1, 0.2), (0.2, 0.3)]
    brain.train(examples, epochs=1, validation_examples=examples)
    assert brain.super_evo_controller.history

