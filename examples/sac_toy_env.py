import torch

class SACGridEnv:
    """Simple 1D grid environment for SAC evaluation.

    The agent starts at position 0 in a linear grid and aims to reach
    ``grid_size - 1``. Actions ``0`` and ``1`` move left and right
    respectively. Each step incurs a small negative reward while reaching
    the goal yields ``+1``. The environment operates on both CPU and GPU
    depending on the provided device.
    """

    def __init__(self, grid_size: int = 5, max_steps: int = 20, device: torch.device | None = None) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state: torch.Tensor | None = None
        self.steps = 0

    def reset(self) -> torch.Tensor:
        """Reset environment to the starting state."""
        self.state = torch.tensor(0, device=self.device, dtype=torch.int64)
        self.steps = 0
        return self._get_observation()

    def _get_observation(self) -> torch.Tensor:
        obs = torch.zeros(self.grid_size, device=self.device, dtype=torch.float32)
        obs[int(self.state.item())] = 1.0
        return obs

    def step(self, action: torch.Tensor | int):
        """Apply ``action`` and return ``(obs, reward, done, info)``.

        Parameters
        ----------
        action: torch.Tensor | int
            ``0`` moves left, ``1`` moves right. Values outside this range
            leave the state unchanged.
        """
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        if action == 1:
            self.state = torch.clamp(self.state + 1, 0, self.grid_size - 1)
        elif action == 0:
            self.state = torch.clamp(self.state - 1, 0, self.grid_size - 1)
        self.steps += 1
        done = bool(self.state.item() == self.grid_size - 1 or self.steps >= self.max_steps)
        reward_value = 1.0 if self.state.item() == self.grid_size - 1 else -0.1
        reward = torch.tensor(reward_value, device=self.device, dtype=torch.float32)
        return self._get_observation(), reward, done, {}

    def to(self, device: torch.device) -> "SACGridEnv":
        """Move environment state to ``device``."""
        self.device = device
        if self.state is not None:
            self.state = self.state.to(device)
        return self
