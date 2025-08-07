import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_interface import attach_marble_layer
from marble_autograd import TransparentMarbleLayer
from tests.test_core_functions import minimal_params


def test_attach_layer_keeps_output_and_runs_marble():
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1))
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    hooked = attach_marble_layer(base, brain, after=0)
    inp = torch.tensor([[0.5]])
    expected = base(inp)
    brain.neuronenblitz.global_activation_count = 0
    out = hooked(inp)
    assert torch.allclose(expected, out)
    assert brain.neuronenblitz.global_activation_count > 0


def test_attach_layer_from_file(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    path = tmp_path / "model.pt"
    torch.save(model, path)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    hooked = attach_marble_layer(str(path), brain)
    inp = torch.tensor([[0.2]])
    assert torch.allclose(model(inp), hooked(inp))
    save_path = tmp_path / "hooked.pt"
    torch.save(hooked, save_path)
    assert save_path.exists()


def test_transparent_layer_mix_weight():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    layer = TransparentMarbleLayer(brain, train_in_graph=True).to(device)
    inp = torch.tensor([[0.4, -0.1]], device=device)
    out_identity = layer(inp, mix_weight=0.0)
    out_mixed = layer(inp, mix_weight=0.5)
    assert torch.allclose(out_identity, inp)
    assert not torch.allclose(out_mixed, inp)
    assert out_mixed.device == device
