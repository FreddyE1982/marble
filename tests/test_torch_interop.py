import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from torch_interop import MarbleTorchAdapter, core_to_torch, torch_to_core
from tests.test_core_functions import minimal_params


def test_marble_torch_adapter_forward():
    params = minimal_params()
    core = Core(params)
    adapter = core_to_torch(core)
    x = torch.randn(len(core.neurons), core.rep_size)
    out = adapter(x)
    assert out.shape == (len(core.neurons), core.rep_size)


def test_brain_checkpoint(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, None, save_dir=str(tmp_path))
    ckpt = tmp_path / "ckpt.pkl"
    brain.save_checkpoint(str(ckpt), epoch=3)
    new_brain = Brain(core, nb, None, save_dir=str(tmp_path))
    loaded_epoch = new_brain.load_checkpoint(str(ckpt))
    assert loaded_epoch == 3
    assert isinstance(new_brain.core, Core)


def test_torch_to_core_updates_weights():
    params = minimal_params()
    core = Core(params)
    adapter = core_to_torch(core)
    adapter.w1.data.fill_(1.0)
    adapter.b1.data.fill_(0.5)
    adapter.w2.data.fill_(2.0)
    adapter.b2.data.fill_(0.1)
    torch_to_core(adapter, core)
    import marble_core

    assert (marble_core._W1 == 1.0).all()
    assert (marble_core._B1 == 0.5).all()
    assert (marble_core._W2 == 2.0).all()
    assert (marble_core._B2 == 0.1).all()

