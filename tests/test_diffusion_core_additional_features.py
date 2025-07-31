import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from diffusion_core import DiffusionCore
from marble_core import MemorySystem


def test_diffusion_core_additional_features(tmp_path):
    params = minimal_params()
    params["diffusion_steps"] = 1
    params["activation_output_dir"] = str(tmp_path)
    params["activation_colormap"] = "plasma"
    params["memory_system"] = {
        "long_term_path": str(tmp_path / "lt.pkl"),
        "threshold": 0.5,
        "consolidation_interval": 1,
    }
    params["cwfl"] = {"num_basis": 2}
    params["harmonic"] = {"base_frequency": 1.0, "decay": 0.9}
    params["fractal"] = {"target_dimension": 2.0}

    ms = MemorySystem(long_term_path=params["memory_system"]["long_term_path"],
                      threshold=0.5, consolidation_interval=1)

    core = DiffusionCore(
        params,
        memory_system=ms,
        cwfl_params=params["cwfl"],
        harmonic_params=params["harmonic"],
        fractal_params=params["fractal"],
        activation_output_dir=params["activation_output_dir"],
        activation_colormap=params["activation_colormap"],
    )
    out = core.diffuse(0.1)
    assert isinstance(out, float)
    assert (tmp_path / "diffusion_1.png").exists()
    assert ms.retrieve("diffusion_output") == out
    if core.cwfl is not None:
        assert core.cwfl.history
    if core.harmonic is not None:
        assert core.harmonic.history
    if core.fractal is not None:
        assert core.fractal.history
