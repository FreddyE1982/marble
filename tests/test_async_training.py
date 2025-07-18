import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params
from tqdm import tqdm as std_tqdm


def test_start_training_background():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    # patch tqdm to avoid notebook dependency
    import marble_brain as mb
    mb.tqdm = std_tqdm
    train_examples = [(0.1, 0.2)] * 20
    brain.start_training(train_examples, epochs=2)
    # wait until thread starts
    timeout = time.time() + 5
    while not brain.training_active and time.time() < timeout:
        time.sleep(0.01)
    assert brain.training_active
    brain.wait_for_training()
    assert not brain.training_active


def test_training_and_inference_simultaneous():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    import marble_brain as mb
    mb.tqdm = std_tqdm
    train_examples = [(0.1, 0.2)] * 50
    brain.start_training(train_examples, epochs=3)
    timeout = time.time() + 5
    while not brain.training_active and time.time() < timeout:
        time.sleep(0.01)
    assert brain.training_active
    out, path = nb.dynamic_wander(0.1)
    assert isinstance(out, float)
    assert path
    brain.wait_for_training()
    assert not brain.training_active
