import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm as std_tqdm

from graph_cache import GRAPH_CACHE
from marble_brain import Brain
from marble_core import Core, DataLoader, precompile_simple_mlp
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_precompile_cache_reuse():
    GRAPH_CACHE.enable(True)
    GRAPH_CACHE.clear()
    sample = torch.randn(1, 4)
    precompile_simple_mlp(sample)
    size1 = GRAPH_CACHE.get_cache_size()
    precompile_simple_mlp(sample)
    size2 = GRAPH_CACHE.get_cache_size()
    assert size1 == size2 == 1


def test_brain_initializes_precompilation():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader(), precompile_graphs=True)
    import marble_brain as mb

    mb.tqdm = std_tqdm
    GRAPH_CACHE.clear()
    brain.train([(0.1, 0.2)], epochs=1)
    assert GRAPH_CACHE.get_cache_size() >= 1
