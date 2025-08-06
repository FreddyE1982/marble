import os
import tensor_backend as tb
from marble_core import Core, perform_message_passing
from tests.test_core_functions import minimal_params
from activation_visualization import plot_activation_heatmap


def test_activation_heatmap(tmp_path):
    tb.set_backend("numpy")
    params = minimal_params()
    core = Core(params)
    perform_message_passing(core)
    out_file = tmp_path / "heat.png"
    plot_activation_heatmap(core, out_file, cmap="plasma")
    assert out_file.exists() and out_file.stat().st_size > 0
