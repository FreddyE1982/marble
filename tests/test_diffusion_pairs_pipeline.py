import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_core import DiffusionCore
from diffusion_pairs_pipeline import DiffusionPairsPipeline
from tests.test_core_functions import minimal_params


def test_diffusion_pairs_pipeline_trains(tmp_path):
    params = minimal_params()
    params["diffusion_steps"] = 1
    core = DiffusionCore(params)
    save_path = tmp_path / "model.pkl"
    pipeline = DiffusionPairsPipeline(core, save_path=str(save_path))
    pairs = [(0.0, 0.1), (0.2, 0.3)]
    out_path = pipeline.train(pairs, epochs=1)
    assert out_path == str(save_path)
    assert save_path.is_file()


def test_diffusion_pairs_pipeline_non_numeric(tmp_path):
    params = minimal_params()
    params["diffusion_steps"] = 1
    core = DiffusionCore(params)
    save_path = tmp_path / "model.pkl"
    pipeline = DiffusionPairsPipeline(core, save_path=str(save_path))
    pairs = [("hello", "foo"), ("world", "bar")]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()
