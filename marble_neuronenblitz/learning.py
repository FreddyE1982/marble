"""Reinforcement learning helpers for :class:`Neuronenblitz`.

These functions operate on a ``Neuronenblitz`` instance and are kept
in a separate module to keep :mod:`core` focused on graph dynamics.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core import Neuronenblitz

import numpy as np
import torch
from soft_actor_critic import create_sac_networks


def enable_rl(nb: "Neuronenblitz") -> None:
    """Enable built-in reinforcement learning."""
    nb.rl_enabled = True


def disable_rl(nb: "Neuronenblitz") -> None:
    """Disable built-in reinforcement learning."""
    nb.rl_enabled = False


def enable_sac(
    nb: "Neuronenblitz",
    state_dim: int,
    action_dim: int,
    *,
    device: str | None = None,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    temperature: float = 0.2,
    tune_entropy: bool = True,
) -> None:
    """Attach Soft Actor-Critic networks to ``nb`` on the requested device.

    Parameters
    ----------
    nb:
        Target ``Neuronenblitz`` instance.
    state_dim:
        Number of observed state features.
    action_dim:
        Number of continuous action dimensions.
    device:
        Torch device string.  Defaults to CUDA when available.
    actor_lr / critic_lr:
        Optimiser learning rates for actor and critic.
    temperature:
        Initial entropy temperature controlling exploration.
    tune_entropy:
        If ``True`` the temperature is automatically tuned toward a target
        entropy of ``-action_dim``. When ``False`` the supplied ``temperature``
        is kept fixed.
    """

    actor, critic = create_sac_networks(state_dim, action_dim, device=device)
    actor_device = next(actor.parameters()).device
    nb.sac_actor = actor
    nb.sac_critic = critic
    nb.sac_device = actor_device
    # expose ``device`` attribute on networks for compatibility
    actor.device = actor_device
    critic.device = actor_device
    nb.sac_actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    nb.sac_critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    nb.sac_auto_temperature = tune_entropy
    nb.sac_entropy_history = []
    if tune_entropy:
        nb.sac_target_entropy = -action_dim
        nb.sac_log_alpha = torch.tensor(
            float(np.log(max(temperature, 1e-6))),
            device=actor.device,
            requires_grad=True,
        )
        nb.sac_alpha_opt = torch.optim.Adam([nb.sac_log_alpha], lr=actor_lr)
        nb.sac_alpha = nb.sac_log_alpha.exp().detach()
    else:
        nb.sac_alpha = torch.tensor(temperature, device=actor.device)


def sac_select_action(nb: "Neuronenblitz", state: np.ndarray | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from the SAC actor on the configured device."""
    if not hasattr(nb, "sac_actor"):
        raise RuntimeError("soft actor-critic not enabled")
    state_t = torch.as_tensor(state, dtype=torch.float32, device=nb.sac_device).unsqueeze(0)
    with torch.no_grad():
        action, log_prob = nb.sac_actor(state_t)
    return action.squeeze(0), log_prob.squeeze(0)


def rl_select_action(
    nb: "Neuronenblitz", state: Tuple[int, int], n_actions: int
) -> int:
    """Return an action using epsilon-greedy selection."""
    if not nb.rl_enabled:
        raise RuntimeError("reinforcement learning disabled")
    if random.random() < nb.rl_epsilon:
        return random.randrange(n_actions)
    q_vals = [nb.dynamic_wander(nb.q_encoding(state, a))[0] for a in range(n_actions)]
    return int(np.argmax(q_vals))


def rl_update(
    nb: "Neuronenblitz",
    state: Tuple[int, int],
    action: int,
    reward: float,
    next_state: Tuple[int, int],
    done: bool,
    n_actions: int = 4,
) -> None:
    """Perform a Q-learning update using ``dynamic_wander``."""
    if not nb.rl_enabled:
        return
    next_q = 0.0
    if not done:
        next_q = max(
            nb.dynamic_wander(nb.q_encoding(next_state, a))[0] for a in range(n_actions)
        )
    target = reward + nb.rl_discount * next_q
    nb.train([(nb.q_encoding(state, action), target)], epochs=1)
    nb.rl_epsilon = max(nb.rl_min_epsilon, nb.rl_epsilon * nb.rl_epsilon_decay)


def sac_update(
    nb: "Neuronenblitz",
    state: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    reward: float,
    next_state: np.ndarray | torch.Tensor,
    done: bool,
    *,
    gamma: float = 0.99,
) -> None:
    """Update SAC actor and critic using a single transition."""
    if not hasattr(nb, "sac_actor"):
        raise RuntimeError("soft actor-critic not enabled")

    device = nb.sac_device
    actor, critic = nb.sac_actor, nb.sac_critic
    actor_opt, critic_opt = nb.sac_actor_opt, nb.sac_critic_opt
    alpha = nb.sac_log_alpha.exp() if nb.sac_auto_temperature else nb.sac_alpha
    alpha_detached = alpha.detach()

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    reward_t = torch.as_tensor([reward], dtype=torch.float32, device=device)
    next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    done_t = torch.as_tensor([done], dtype=torch.float32, device=device)

    with torch.no_grad():
        next_action, next_log_prob = actor(next_state_t)
        q1_next, q2_next = critic(next_state_t, next_action)
        q_target = reward_t + (1 - done_t) * gamma * (
            torch.min(q1_next, q2_next) - alpha_detached * next_log_prob
        )

    q1, q2 = critic(state_t, action_t)
    critic_loss = torch.nn.functional.mse_loss(q1, q_target) + torch.nn.functional.mse_loss(q2, q_target)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    new_action, log_prob = actor(state_t)
    q1_new, q2_new = critic(state_t, new_action)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha_detached * log_prob - q_new).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    entropy = (-log_prob).mean().detach().cpu().item()
    nb.sac_entropy_history.append(entropy)

    if nb.sac_auto_temperature:
        alpha_loss = -(nb.sac_log_alpha * (log_prob + nb.sac_target_entropy).detach()).mean()
        nb.sac_alpha_opt.zero_grad()
        alpha_loss.backward()
        nb.sac_alpha_opt.step()
        nb.sac_alpha = nb.sac_log_alpha.exp().detach()


def plot_sac_entropy(nb: "Neuronenblitz", file_path: str | Path) -> None:
    """Plot recorded policy entropy and save to ``file_path``.

    Parameters
    ----------
    nb:
        Target ``Neuronenblitz`` instance with SAC enabled.
    file_path:
        Destination path for the saved plot image.
    """
    if not hasattr(nb, "sac_entropy_history"):
        raise RuntimeError("soft actor-critic not enabled")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(nb.sac_entropy_history)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Policy entropy")
    ax.set_title("SAC Policy Entropy")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
