import random

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


def tiny_params():
    return {
        "width": 2,
        "height": 2,
        "max_iter": 1,
        "representation_size": 4,
        "vram_limit_mb": 0.1,
        "ram_limit_mb": 0.1,
        "disk_limit_mb": 0.1,
        "random_seed": 0,
    }


def test_bidirectional_attachment():
    random.seed(0)
    core = Core(tiny_params())
    nb = Neuronenblitz(core)
    assert core.neuronenblitz is nb
    assert nb.core is core


def test_attach_core_switch():
    random.seed(0)
    core1 = Core(tiny_params())
    nb = Neuronenblitz(core1)
    core2 = Core(tiny_params())
    nb.attach_core(core2)
    assert nb.core is core2
    assert core2.neuronenblitz is nb
    assert core1.neuronenblitz is None


def test_detach_methods():
    random.seed(0)
    core = Core(tiny_params())
    nb = Neuronenblitz(core)
    core.detach_neuronenblitz()
    assert core.neuronenblitz is None
    assert nb.core is None
    nb.attach_core(core)
    nb.detach_core()
    assert core.neuronenblitz is None
    assert nb.core is None
