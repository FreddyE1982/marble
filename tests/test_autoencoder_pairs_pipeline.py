import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bit_tensor_dataset import BitTensorDataset
from marble_core import Core
from autoencoder_pairs_pipeline import AutoencoderPairsPipeline
from tests.test_core_functions import minimal_params


def test_autoencoder_pairs_pipeline_trains(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "ae.pkl"
    pipeline = AutoencoderPairsPipeline(core, save_path=str(save_path))
    values = [0.1, 0.2, 0.3]
    pipeline.train(values, epochs=1)
    assert save_path.is_file()


def test_autoencoder_pairs_pipeline_bit_dataset(tmp_path):
    params = minimal_params()
    core = Core(params)
    save_path = tmp_path / "ae_bit.pkl"
    pipeline = AutoencoderPairsPipeline(core, save_path=str(save_path))
    pairs = [(0.1, 0.1), (0.2, 0.2)]
    ds = BitTensorDataset(pairs)
    pipeline.train(ds, epochs=1)
    assert save_path.is_file()
