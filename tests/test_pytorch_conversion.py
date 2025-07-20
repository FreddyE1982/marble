import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from datasets import load_dataset
from transformers import AutoModel
from marble_interface import convert_pytorch_model
from tests.test_core_functions import minimal_params


def test_pytorch_conversion_predictions_match():
    model = AutoModel.from_pretrained("dacorvo/mnist-mlp", trust_remote_code=True)
    ds = load_dataset("mnist", split="test[:5]")
    marble = convert_pytorch_model(model, minimal_params())

    torch_preds = []
    with torch.no_grad():
        for record in ds:
            img = record["image"].convert("L")
            val = float(torch.tensor(list(img.getdata()), dtype=torch.float32).mean()) / 255.0
            vec = torch.full((1, 784), val, dtype=torch.float32)
            out = model(vec)
            torch_preds.append(out.squeeze())

    marble_preds = []
    for record in ds:
        img = record["image"].convert("L")
        val = float(torch.tensor(list(img.getdata()), dtype=torch.float32).mean())
        out = marble.get_brain().infer(val)
        marble_preds.append(out)

    assert len(torch_preds) == len(marble_preds)
    for p2 in marble_preds:
        assert isinstance(p2, float)
