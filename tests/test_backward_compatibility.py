import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core
from marble_utils import core_from_json, core_to_json
from tests.test_core_functions import minimal_params


def test_core_from_json_backward_compat():
    params = minimal_params()
    core = Core(params)
    json_str = core_to_json(core)
    data = json.loads(json_str)
    for syn in data.get("synapses", []):
        syn.pop("synapse_type", None)
        syn.pop("potential", None)
    old_json = json.dumps(data)
    new_core = core_from_json(old_json)
    assert len(new_core.neurons) == len(core.neurons)
    assert len(new_core.synapses) == len(core.synapses)
