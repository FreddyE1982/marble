import random
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
from tqdm import tqdm as std_tqdm


def test_early_stopping_triggers():
    random.seed(0)
    marble_brain.tqdm = std_tqdm
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = marble_brain.Brain(
        core,
        nb,
        DataLoader(),
        early_stop_enabled=True,
        early_stopping_patience=1,
        early_stopping_delta=0.0,
    )
    examples = [(0.1, 0.1)]
    brain.train(examples, epochs=5, validation_examples=examples)
    assert len(brain.meta_controller.loss_history) < 5
