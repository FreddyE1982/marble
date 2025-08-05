import torch
from bit_tensor_dataset import BitTensorDataset
from highlevel_pipeline import HighLevelPipeline
from marble_neuronenblitz import Neuronenblitz
from marble_core import Core


def test_highlevel_pipeline_trains_on_streamed_shards():
    pairs = [(i, i + 1) for i in range(8)]
    dataset = BitTensorDataset(pairs, device="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = HighLevelPipeline()
    hp.add_step(
        "streaming_dataset_step",
        module="marble_interface",
        params={"dataset": dataset, "batch_size": 4, "prefetch": 2, "device": device},
    )
    nb = Neuronenblitz(Core({}))
    _, results = hp.execute(nb)
    assert len(nb.training_history) > 0
    assert isinstance(results[0], list)
