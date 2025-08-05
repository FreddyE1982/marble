import torch
from bit_tensor_dataset import BitTensorDataset
from pipeline import Pipeline
import marble_interface
from marble_neuronenblitz import Neuronenblitz
from marble_core import Core


def test_streamed_training_runs():
    pairs = [(i, i + 1) for i in range(8)]
    dataset = BitTensorDataset(pairs, device="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Pipeline([
        {
            "func": "streaming_dataset_step",
            "module": "marble_interface",
            "params": {"dataset": dataset, "batch_size": 4, "prefetch": 2, "device": device},
        }
    ])
    nb = Neuronenblitz(Core({}))
    pipe.execute(nb)
    assert len(nb.training_history) > 0
