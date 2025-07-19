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
            flat = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
            out = model(flat.unsqueeze(0))
            torch_preds.append(out.squeeze())

    marble_preds = []
    for record in ds:
        img = record["image"].convert("L")
        val = float(torch.tensor(list(img.getdata()), dtype=torch.float32).mean())
        out = marble.get_brain().infer(val)
        marble_preds.append(out)

    for p1, p2 in zip(torch_preds, marble_preds):
        assert torch.allclose(p1.float(), torch.tensor(p2).float(), atol=1e-5)
