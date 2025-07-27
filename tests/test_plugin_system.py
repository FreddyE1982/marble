import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plugin_system import load_plugins
from marble_core import NEURON_TYPES, SYNAPSE_TYPES


def test_plugin_registration(tmp_path):
    plugin_code = (
        "def register(reg_neuron, reg_synapse):\n"
        "    reg_neuron('test_neuron')\n"
        "    reg_synapse('test_synapse')\n"
    )
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(plugin_code)

    load_plugins([str(tmp_path)])

    assert "test_neuron" in NEURON_TYPES
    assert "test_synapse" in SYNAPSE_TYPES
