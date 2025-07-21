import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from marble_core import Core
from marble_utils import core_to_json, core_from_json
from system_metrics import get_system_memory_usage, get_gpu_memory_usage
from tests.test_core_functions import minimal_params


def test_core_json_roundtrip():
    params = minimal_params()
    core = Core(params)
    json_str = core_to_json(core)
    assert isinstance(json_str, str)
    new_core = core_from_json(json_str)
    assert len(new_core.neurons) == len(core.neurons)
    assert len(new_core.synapses) == len(core.synapses)
    # Compare a few attributes
    assert new_core.neurons[0].value == core.neurons[0].value


def test_memory_usage_functions():
    ram = get_system_memory_usage()
    gpu = get_gpu_memory_usage()
    assert isinstance(ram, float) and ram > 0
    assert isinstance(gpu, float)
