import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from diffusion_core import DiffusionCore
from marble_core import NEURON_TYPES


def test_diffusion_core_predictive_and_schema(tmp_path):
    params = minimal_params()
    params["diffusion_steps"] = 1
    plugin_file = tmp_path / "myplugin.py"
    plugin_file.write_text("def register(reg_neuron, reg_synapse):\n    reg_neuron('schema_neuron')\n")

    core = DiffusionCore(
        params,
        predictive_coding_params={"num_layers": 1, "latent_dim": 4, "learning_rate": 0.001},
        schema_induction_params={"support_threshold": 1, "max_schema_size": 2},
        plugin_dirs=[str(tmp_path)],
    )

    initial = len(core.neurons)
    out = core.diffuse(0.2)
    assert isinstance(out, float)
    assert "schema_neuron" in NEURON_TYPES
    assert len(core.neurons) > initial
    assert core.predictive_coding is not None

