import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import global_workspace
from neuromodulatory_system import NeuromodulatorySystem
from diffusion_core import DiffusionCore
from tests.test_core_functions import minimal_params


def test_diffusion_core_workspace_broadcast():
    params = minimal_params()
    params["diffusion_steps"] = 1
    params["workspace_broadcast"] = True
    gw = global_workspace.activate(capacity=1)
    ns = NeuromodulatorySystem()
    core = DiffusionCore(params, neuromodulatory_system=ns)
    out = core.diffuse(0.0)
    assert isinstance(out, float)
    assert gw.queue and gw.queue[-1].content == out

