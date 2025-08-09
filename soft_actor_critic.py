import torch
import torch.nn as nn


def _default_device(device: str | None = None) -> torch.device:
    """Return torch device preferring CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Gaussian policy network with Tanh squashing.

    Parameters
    ----------
    state_dim:
        Number of state features.
    action_dim:
        Number of continuous actions.
    hidden_dim:
        Size of hidden layers.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action and its log-probability for ``state``."""
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """Twin Q-network used by Soft Actor-Critic.

    Parameters
    ----------
    state_dim:
        Number of state features.
    action_dim:
        Number of continuous actions.
    hidden_dim:
        Size of hidden layers.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Q-value estimates for ``(state, action)``."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


def create_sac_networks(
    state_dim: int,
    action_dim: int,
    *,
    hidden_dim: int = 256,
    device: str | None = None,
) -> tuple[Actor, Critic]:
    """Construct actor and critic networks on the requested device."""
    device_t = _default_device(device)
    actor = Actor(state_dim, action_dim, hidden_dim).to(device_t)
    critic = Critic(state_dim, action_dim, hidden_dim).to(device_t)
    return actor, critic


__all__ = ["Actor", "Critic", "create_sac_networks"]
