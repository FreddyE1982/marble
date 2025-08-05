import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn as nn

from marble_core import LOSS_MODULES, NEURON_TYPES, SYNAPSE_TYPES
from plugin_system import load_plugins


def test_plugin_registration(tmp_path):
    plugin_code = (
        "import torch.nn as nn\n"
        "class MyLoss(nn.Module):\n"
        "    def forward(self, pred, target):\n"
        "        return nn.functional.l1_loss(pred, target)\n"
        "def register(reg_neuron, reg_synapse, reg_loss):\n"
        "    reg_neuron('test_neuron')\n"
        "    reg_synapse('test_synapse')\n"
        "    reg_loss('test_loss', MyLoss)\n"
    )
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(plugin_code)

    legacy_code = (
        "def register(reg_neuron, reg_synapse):\n"
        "    reg_neuron('legacy_neuron')\n"
        "    reg_synapse('legacy_synapse')\n"
    )
    legacy_file = tmp_path / "legacy_plugin.py"
    legacy_file.write_text(legacy_code)

    load_plugins([str(tmp_path)])

    assert "test_neuron" in NEURON_TYPES
    assert "test_synapse" in SYNAPSE_TYPES
    assert "test_loss" in LOSS_MODULES
    assert issubclass(LOSS_MODULES["test_loss"], nn.Module)
    assert "legacy_neuron" in NEURON_TYPES
    assert "legacy_synapse" in SYNAPSE_TYPES
