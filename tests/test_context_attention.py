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


def test_causal_mask_blocks_future_tokens():
    attn = ContextAwareAttention(2, causal=True)
    q = torch.randn(1, 3, 2)
    k = torch.randn(1, 3, 2)
    v = torch.randn(1, 3, 2)
    _, weights = attn(q, k, v, return_weights=True)
    assert torch.allclose(weights.triu(1), torch.zeros_like(weights.triu(1)))


def test_gating_layer_bounds():
    attn = ContextAwareAttention(
        2, gating={"enabled": True, "mode": "sine", "frequency": 1.0}
    )
    gate = attn.gating(5)
    assert gate.min() >= -1.0 and gate.max() <= 1.0
    attn = ContextAwareAttention(
        2, gating={"enabled": True, "mode": "chaos", "chaos": 3.7}
    )
    gate = attn.gating(5)
    assert gate.min() >= 0.0 and gate.max() <= 1.0
