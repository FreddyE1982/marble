import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from pipeline import Pipeline
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from synthetic_dataset import generate_linear_dataset
from tests.test_core_functions import minimal_params


def tensor_dataset():
    return [
        (torch.tensor(0.0), torch.tensor(0.0)),
        (torch.tensor(1.0), torch.tensor(1.0)),
    ]


def test_dataset_step_detection():
    pipe = Pipeline([
        {"module": "synthetic_dataset", "func": "generate_linear_dataset"},
        {"func": "noop"},
    ])
    idxs = pipe._dataset_step_indices()
    assert idxs == [0]


def test_auto_training_loop_cpu():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    steps = [
        {
            "module": "synthetic_dataset",
            "func": "generate_linear_dataset",
            "params": {"n_samples": 5},
        }
    ]
    pipe = Pipeline(steps)
    pipe.execute(marble=nb)
    assert len(nb.get_training_history()) > 0


def test_auto_training_loop_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    steps = [
        {"module": "tests.test_auto_nb_training_loop", "func": "tensor_dataset"}
    ]
    pipe = Pipeline(steps)
    pipe.execute(marble=nb)
    assert len(nb.get_training_history()) > 0
