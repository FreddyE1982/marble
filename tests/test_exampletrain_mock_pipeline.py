import torch

from exampletrain_utils import MockStableDiffusionPipeline


def test_mock_pipeline_cpu_gpu_parity():
    """Mock Stable Diffusion pipeline should behave identically on CPU and GPU."""
    caption = "synthetic caption"

    cpu_pipe = MockStableDiffusionPipeline(device="cpu")
    cpu_tokens = cpu_pipe.tokenizer(caption, return_tensors="pt")
    cpu_emb = cpu_pipe.text_encoder(**cpu_tokens).last_hidden_state

    if torch.cuda.is_available():
        gpu_pipe = MockStableDiffusionPipeline(device="cuda")
        gpu_tokens = gpu_pipe.tokenizer(caption, return_tensors="pt")
        gpu_emb = gpu_pipe.text_encoder(**gpu_tokens).last_hidden_state

        assert torch.allclose(cpu_tokens["input_ids"], gpu_tokens["input_ids"].cpu())
        assert torch.allclose(cpu_emb, gpu_emb.cpu())
    else:
        assert cpu_tokens["input_ids"].device.type == "cpu"
        assert cpu_emb.device.type == "cpu"
