import random
import numpy as np
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


class GridWorld:
    """Simple grid environment with deterministic transitions."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.reset()

    @property
    def n_actions(self) -> int:
        return 4  # up, down, left, right

    def reset(self) -> tuple[int, int]:
        self.pos = self.start
        return self.pos

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        x, y = self.pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.size - 1:
            y += 1
        self.pos = (x, y)
        done = self.pos == self.goal
        reward = 10.0 if done else -1.0
        return self.pos, reward, done


class MarbleQLearningAgent:
    """Q-learning agent powered by the MARBLE system."""

    def __init__(
        self,
        core: Core,
        nb: Neuronenblitz,
        discount: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.95,
        min_epsilon: float = 0.1,
        double_q: bool = False,
    ) -> None:
        self.core = core
        self.nb = nb
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.double_q = double_q
        if double_q:
            self.q_table_a: dict[tuple[tuple[int, int], int], float] = {}
            self.q_table_b: dict[tuple[tuple[int, int], int], float] = {}
        else:
            self.q_table: dict[tuple[tuple[int, int], int], float] = {}

    def _encode(self, state: tuple[int, int], action: int) -> float:
        """Encode state-action pair into a numeric input."""
        return float(state[0] * 10 + state[1] + action / 10)

    def select_action(self, state: tuple[int, int], n_actions: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(n_actions)
        if self.double_q:
            q_values = [
                (
                    self.q_table_a.get((state, a), 0.0)
                    + self.q_table_b.get((state, a), 0.0)
                )
                / 2
                for a in range(n_actions)
            ]
        else:
            q_values = [
                self.q_table.get((state, a), 0.0) for a in range(n_actions)
            ]
        return int(np.argmax(q_values))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        if self.double_q:
            tables = [(self.q_table_a, self.q_table_b), (self.q_table_b, self.q_table_a)]
            update_table, eval_table = random.choice(tables)
            if not done:
                best_a = max(
                    range(4),
                    key=lambda a: update_table.get((next_state, a), 0.0),
                )
                next_q = eval_table.get((next_state, best_a), 0.0)
            else:
                next_q = 0.0
            current = update_table.get((state, action), 0.0)
            target = reward + self.discount * next_q
            update_table[(state, action)] = current + 0.1 * (target - current)
        else:
            next_q = 0.0
            if not done:
                next_q = max(
                    self.q_table.get((next_state, a), 0.0) for a in range(4)
                )
            current = self.q_table.get((state, action), 0.0)
            target = reward + self.discount * next_q
            self.q_table[(state, action)] = current + 0.1 * (target - current)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_gridworld(
    agent: MarbleQLearningAgent,
    env: GridWorld,
    episodes: int,
    max_steps: int = 50,
    seed: int | None = 0,
) -> list[float]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rewards = []
    for episode in range(episodes):
        agent.epsilon = 0.0 if episode == episodes - 1 else 1.0
        state = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, env.n_actions)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
            if done:
                break
        rewards.append(total)
    return rewards
