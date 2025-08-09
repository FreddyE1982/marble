import pytest
import torch

from soft_actor_critic import create_sac_networks


def _generate_sample(state_dim: int, batch: int = 3):
    """Generate a random state tensor for tests."""
    return torch.randn(batch, state_dim)


def test_create_sac_networks_cpu():
    """Actor and critic should operate on CPU and support backpropagation."""
    torch.manual_seed(0)
    actor, critic = create_sac_networks(4, 2, device="cpu")
    state = _generate_sample(4)
    action, log_prob = actor(state)
    q1, q2 = critic(state, action)
    loss = (q1 + q2 + log_prob).mean()
    loss.backward()
    assert action.shape == (3, 2)
    assert log_prob.shape == (3, 1)
    assert q1.shape == (3, 1)
    assert q2.shape == (3, 1)
    for p in list(actor.parameters()) + list(critic.parameters()):
        assert p.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_create_sac_networks_cpu_gpu_parity():
    """CPU and GPU networks with same seed produce identical outputs and gradients."""
    torch.manual_seed(0)
    actor_cpu, critic_cpu = create_sac_networks(4, 2, device="cpu")
    torch.manual_seed(0)
    actor_gpu, critic_gpu = create_sac_networks(4, 2, device="cuda")

    state = _generate_sample(4)
    action_cpu, log_prob_cpu = actor_cpu(state)
    q1_cpu, q2_cpu = critic_cpu(state, action_cpu)
    loss_cpu = (q1_cpu + q2_cpu + log_prob_cpu).mean()
    loss_cpu.backward()

    state_gpu = state.to("cuda")
    action_gpu, log_prob_gpu = actor_gpu(state_gpu)
    q1_gpu, q2_gpu = critic_gpu(state_gpu, action_gpu)
    loss_gpu = (q1_gpu + q2_gpu + log_prob_gpu).mean()
    loss_gpu.backward()

    assert torch.allclose(action_cpu, action_gpu.cpu(), atol=1e-6)
    assert torch.allclose(log_prob_cpu, log_prob_gpu.cpu(), atol=1e-6)
    assert torch.allclose(q1_cpu, q1_gpu.cpu(), atol=1e-6)
    assert torch.allclose(q2_cpu, q2_gpu.cpu(), atol=1e-6)
    for p_cpu, p_gpu in zip(actor_cpu.parameters(), actor_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-6)
    for p_cpu, p_gpu in zip(critic_cpu.parameters(), critic_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-6)
