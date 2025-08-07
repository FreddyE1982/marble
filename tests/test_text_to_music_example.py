import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
import torch

from bit_tensor_streaming_dataset import BitTensorStreamingDataset
from highlevel_pipeline import HighLevelPipeline


def _dummy_dataset():
    data = [{
        "lyrics": "la la",
        "prompt": "happy",
        "tags": ["test"],
        "duration": 1.0,
        "audio": [0.0, 0.1, 0.2, 0.3],
    }]
    ds = Dataset.from_list(data)

    def format_record(rec):
        return {
            "input": {
                "lyrics": rec["lyrics"],
                "prompt": rec["prompt"],
                "tags": rec["tags"],
                "duration": rec["duration"],
            },
            "target": rec["audio"],
        }

    return ds.map(format_record, remove_columns=ds.column_names)


def test_streaming_wrapper_encodes_tensor():
    ds = _dummy_dataset()
    stream = BitTensorStreamingDataset(ds)
    inp, tgt = next(iter(stream))
    assert isinstance(inp, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)


def test_pipeline_executes_with_stream():
    ds = _dummy_dataset()
    stream = BitTensorStreamingDataset(ds)
    pipeline = (
        HighLevelPipeline()
        .marble_interface.new_marble_system()
        .marble_interface.train_marble_system(train_examples=stream, epochs=1)
    )
    pipeline.execute()
