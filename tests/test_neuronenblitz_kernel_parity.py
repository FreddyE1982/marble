import math
import pytest
import torch

from neuronenblitz_kernel import apply_weight_updates


def _apply_weight_updates_cpu(
    source: torch.Tensor,
    weights: torch.Tensor,
    potentials: torch.Tensor,
    momentum: torch.Tensor,
    grad_sq: torch.Tensor,
    prev_grad: torch.Tensor,
    eligibility: torch.Tensor,
    mem_gate: torch.Tensor,
    error: float,
    learning_rate: float,
    momentum_coeff: float,
    rms_beta: float,
    grad_epsilon: float,
    cap: float,
    weight_limit: float,
    gradient_score_scale: float,
    synapse_potential_cap: float,
    path_len: int,
):
    """Reference implementation of the CUDA kernel in pure PyTorch."""
    dtype = weights.dtype
    src = source.float()
    w = weights.float().clone()
    pot = potentials.float().clone()
    mom = momentum.float().clone()
    gs = grad_sq.float().clone()
    pg = prev_grad.float().clone()
    elig = eligibility.float()
    mem = mem_gate.float()
    delta = (error * src) / float(path_len + 1)
    flip_mask = pg * delta < 0.0
    delta = torch.where(flip_mask, delta * 0.5, delta)
    pg = delta.clone()
    delta = delta * elig
    v = rms_beta * gs + (1.0 - rms_beta) * (delta * delta)
    gs = v
    scaled_delta = delta / torch.sqrt(v + grad_epsilon)
    m = momentum_coeff * mom + scaled_delta
    mom = m
    update = learning_rate * (momentum_coeff * m + scaled_delta)
    update = torch.clamp(update, -cap, cap)
    w = torch.clamp(w + update, -weight_limit, weight_limit)
    pot = torch.minimum(
        pot + torch.abs(scaled_delta) * gradient_score_scale,
        torch.tensor(synapse_potential_cap, dtype=pot.dtype),
    )
    mem_factor = 1.0 + mem
    scores = torch.abs(torch.tensor(error, dtype=pot.dtype)) * torch.abs(w) / float(path_len)
    scores = scores * mem_factor
    return (
        w.to(dtype),
        pot.to(dtype),
        mom.to(dtype),
        gs.to(dtype),
        pg.to(dtype),
        scores.to(dtype),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_apply_weight_updates_cpu_gpu_parity(dtype):
    torch.manual_seed(0)
    size = 64
    device = "cuda"
    kwargs = dict(device=device, dtype=dtype)
    source = torch.randn(size, **kwargs)
    weights = torch.randn(size, **kwargs)
    potentials = torch.zeros(size, **kwargs)
    momentum = torch.zeros(size, **kwargs)
    grad_sq = torch.ones(size, **kwargs)
    prev_grad = torch.zeros(size, **kwargs)
    eligibility = torch.ones(size, **kwargs)
    mem_gate = torch.zeros(size, **kwargs)
    params = dict(
        error=0.1,
        learning_rate=0.01,
        momentum_coeff=0.9,
        rms_beta=0.99,
        grad_epsilon=1e-8,
        cap=0.1,
        weight_limit=1.0,
        gradient_score_scale=0.5,
        synapse_potential_cap=10.0,
        path_len=10,
    )

    w_cpu, pot_cpu, mom_cpu, gs_cpu, pg_cpu, sc_cpu = _apply_weight_updates_cpu(
        source.cpu(),
        weights.cpu(),
        potentials.cpu(),
        momentum.cpu(),
        grad_sq.cpu(),
        prev_grad.cpu(),
        eligibility.cpu(),
        mem_gate.cpu(),
        **params,
    )

    w_gpu, pot_gpu, mom_gpu, gs_gpu, pg_gpu, sc_gpu = apply_weight_updates(
        source,
        weights.clone(),
        potentials.clone(),
        momentum.clone(),
        grad_sq.clone(),
        prev_grad.clone(),
        eligibility,
        mem_gate,
        **params,
    )

    atol = 1e-3 if dtype == torch.float16 else 1e-6
    assert torch.allclose(w_cpu, w_gpu.cpu(), atol=atol)
    assert torch.allclose(pot_cpu, pot_gpu.cpu(), atol=atol)
    assert torch.allclose(mom_cpu, mom_gpu.cpu(), atol=atol)
    assert torch.allclose(gs_cpu, gs_gpu.cpu(), atol=atol)
    assert torch.allclose(pg_cpu, pg_gpu.cpu(), atol=atol)
    assert torch.allclose(sc_cpu, sc_gpu.cpu(), atol=atol)

