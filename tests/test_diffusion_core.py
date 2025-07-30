import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from diffusion_core import DiffusionCore


def test_diffusion_core_runs():
    params = minimal_params()
    params["diffusion_steps"] = 3
    core = DiffusionCore(params)
    out = core.diffuse(0.0)
    assert isinstance(out, float)
    assert core.history
