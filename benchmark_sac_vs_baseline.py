import torch
from examples.sac_toy_env import SACGridEnv
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.learning import enable_sac, sac_select_action, sac_update


def run_baseline(env: SACGridEnv, episodes: int = 100) -> float:
    """Run random baseline wanderer and return average steps to goal."""
    steps_list: list[int] = []
    for _ in range(episodes):
        env.reset()
        done = False
        steps = 0
        while not done:
            action = torch.randint(0, 2, (1,), device=env.device)
            _, _, done, _ = env.step(action)
            steps += 1
        steps_list.append(steps)
    return sum(steps_list) / len(steps_list)


def run_sac(env: SACGridEnv, episodes: int = 100) -> float:
    """Train SAC-enabled ``Neuronenblitz`` and return average steps to goal."""
    core = Core({"width": 1, "height": 1})
    nb = Neuronenblitz(core)
    enable_sac(nb, state_dim=env.grid_size, action_dim=1, device=str(env.device))
    steps_list: list[int] = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action_t, _ = sac_select_action(nb, state)
            env_action = 1 if action_t.item() > 0 else 0
            next_state, reward, done, _ = env.step(env_action)
            sac_update(nb, state, action_t, float(reward.item()), next_state, done)
            state = next_state
            steps += 1
        steps_list.append(steps)
    return sum(steps_list) / len(steps_list)


def benchmark(episodes: int = 100, grid_size: int = 5, max_steps: int = 20) -> None:
    """Compare baseline and SAC wanderers on ``SACGridEnv``.

    Runs both strategies for ``episodes`` episodes and prints average steps
    required to reach the goal on CPU or GPU depending on availability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SACGridEnv(grid_size=grid_size, max_steps=max_steps, device=device)
    baseline_steps = run_baseline(env, episodes)
    sac_steps = run_sac(env, episodes)
    improvement = ((baseline_steps - sac_steps) / baseline_steps) * 100 if baseline_steps else 0.0
    print(f"Baseline avg steps: {baseline_steps:.2f}")
    print(f"SAC avg steps: {sac_steps:.2f}")
    print(f"SAC reduces steps by {improvement:.1f}%")


if __name__ == "__main__":
    benchmark()
