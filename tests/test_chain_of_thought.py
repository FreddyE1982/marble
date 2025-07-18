import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_generate_chain_of_thought():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    output, chain = brain.generate_chain_of_thought(0.1)
    assert isinstance(output, float)
    assert isinstance(chain, list) and chain
    first = chain[0]
    assert {"from", "to", "weight", "input", "output"}.issubset(first.keys())
