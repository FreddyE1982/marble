import math
import random
from marble_core import Core
from tests.test_core_functions import minimal_params


def _make_core(init_type):
    p = minimal_params()
    p["weight_init_type"] = init_type
    p["weight_init_min"] = -0.5
    p["weight_init_max"] = 0.5
    return Core(p)


def test_xavier_normal_range():
    core = _make_core("xavier_normal")
    w = core._init_weight(fan_in=3, fan_out=4)
    std = math.sqrt(2.0 / (3 + 4))
    assert abs(w) < 4 * std


def test_kaiming_uniform_range():
    core = _make_core("kaiming_uniform")
    w = core._init_weight(fan_in=5)
    limit = math.sqrt(6.0 / 5)
    assert -limit <= w <= limit
