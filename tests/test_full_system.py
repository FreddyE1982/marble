import random
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import marble_main
import marble_imports
import marble_brain
from marble_main import MARBLE
from marble_base import MetricsVisualizer
from tests.test_core_functions import minimal_params
from tqdm import tqdm as std_tqdm


def test_full_system_workflow():
    random.seed(0)
    # Patch modules for test environment
    marble_main.MetricsVisualizer = MetricsVisualizer
    marble_imports.tqdm = std_tqdm
    marble_brain.tqdm = std_tqdm

    params = minimal_params()
    marble_system = MARBLE(params, formula=None, formula_num_neurons=5)

    train_examples = [(0.1, 0.2), (0.3, 0.4)]
    val_examples = [(0.5, 0.5)]

    marble_system.get_brain().train(train_examples, epochs=1, validation_examples=val_examples)
    val_loss = marble_system.get_brain().validate(val_examples)

    output, path = marble_system.get_neuronenblitz().dynamic_wander(0.1)

    assert isinstance(val_loss, float)
    assert isinstance(output, float)
    assert isinstance(path, list) and path
