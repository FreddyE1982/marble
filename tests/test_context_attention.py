from marble_neuronenblitz.context_attention import ContextAwareAttention
import torch


def test_context_attention_cpu():
    attn = ContextAwareAttention(4)
    q = torch.randn(1, 4)
    k = torch.randn(1, 4)
    v = torch.randn(1, 4)
    out = attn(q, k, v)
    assert out.shape == (1, 4)


def test_context_attention_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = ContextAwareAttention(2).to(device)
    q = torch.randn(1, 2, device=device)
    k = torch.randn(1, 2, device=device)
    v = torch.randn(1, 2, device=device)
    out = attn(q, k, v)
    assert out.device.type == device
