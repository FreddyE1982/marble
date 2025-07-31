import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bit_tensor_dataset import BitTensorDataset
from marble_core import Core
from continual_pairs_pipeline import ContinualPairsPipeline
from tests.test_core_functions import minimal_params


def test_continual_pairs_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "cont.pkl"
    pipeline = ContinualPairsPipeline(core, save_path=str(save_path))
    pairs = [(0.1, 0.2), (0.3, 0.4)]
    pipeline.train(pairs, epochs=1)
    assert save_path.is_file()


def test_continual_pairs_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "cont_bit.pkl"
    pipeline = ContinualPairsPipeline(core, save_path=str(save_path))
    pairs = [(0.1, 0.2), (0.2, 0.3)]
    ds = BitTensorDataset(pairs)
    pipeline.train(ds, epochs=1)
    assert save_path.is_file()
