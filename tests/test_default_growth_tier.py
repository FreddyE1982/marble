import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_brain import Brain
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz


def minimal_params():
    return {
        "xmin": -2.0,
        "xmax": 1.0,
        "ymin": -1.5,
        "ymax": 1.5,
        "width": 3,
        "height": 3,
        "max_iter": 5,
        "representation_size": 4,
        "message_passing_alpha": 0.5,
        "vram_limit_mb": 0.1,
        "ram_limit_mb": 0.1,
        "disk_limit_mb": 0.1,
        "random_seed": 0,
        "attention_temperature": 1.0,
        "attention_dropout": 0.0,
        "representation_noise_std": 0.0,
        "weight_init_type": "uniform",
        "weight_init_std": 1.0,
    }


def test_core_uses_default_growth_tier():
    params = minimal_params()
    params["default_growth_tier"] = "ram"
    core = Core(params)
    assert core.choose_new_tier() == "ram"


def test_brain_uses_default_growth_tier():
    params = minimal_params()
    params["default_growth_tier"] = "ram"
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    assert brain.choose_growth_tier() == "ram"
