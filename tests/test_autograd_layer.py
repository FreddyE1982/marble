import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from marble_autograd import MarbleAutogradLayer
from tests.test_core_functions import minimal_params


def test_autograd_forward_backward():
    torch.manual_seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    layer = MarbleAutogradLayer(brain, learning_rate=0.1)
    inp = torch.tensor(0.5, requires_grad=True)
    out = layer(inp)
    loss = (out - 1.0) ** 2
    before = [s.weight for s in core.synapses]
    loss.backward()
    changed = any(abs(syn.weight - before[i]) > 1e-6 for i, syn in enumerate(core.synapses))
    assert changed


def test_autograd_gradient_accumulation():
    torch.manual_seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    layer = MarbleAutogradLayer(brain, learning_rate=0.1, accumulation_steps=2)
    inp = torch.tensor(0.5, requires_grad=True)
    out = layer(inp)
    loss = (out - 1.0) ** 2
    before = [s.weight for s in core.synapses]
    loss.backward()
    changed = any(abs(syn.weight - before[i]) > 1e-6 for i, syn in enumerate(core.synapses))
    assert not changed
    inp2 = torch.tensor(0.5, requires_grad=True)
    out2 = layer(inp2)
    loss2 = (out2 - 1.0) ** 2
    loss2.backward()
    changed = any(abs(syn.weight - before[i]) > 1e-6 for i, syn in enumerate(core.synapses))
    assert changed

def test_autograd_scheduler():
    import torch
    from marble_core import Core, DataLoader
    from marble_neuronenblitz import Neuronenblitz
    from marble_brain import Brain

    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    def sched(step):
        return 0.1 / (step + 1)
    layer = MarbleAutogradLayer(brain, learning_rate=0.1, accumulation_steps=2, scheduler=sched)
    inp = torch.tensor(0.5, requires_grad=True)
    out = layer(inp)
    loss = (out - 1.0) ** 2
    loss.backward()
    lr_before = layer.learning_rate
    inp2 = torch.tensor(0.5, requires_grad=True)
    out2 = layer(inp2)
    loss2 = (out2 - 1.0) ** 2
    loss2.backward()
    assert layer.learning_rate < lr_before
