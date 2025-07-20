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
    ) -> None:
        self.core = core
        self.nb = nb
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def _encode(self, state: tuple[int, int], action: int) -> float:
        """Encode state-action pair into a numeric input."""
        return float(state[0] * 10 + state[1] + action / 10)

    def select_action(self, state: tuple[int, int], n_actions: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(n_actions)
        q_values = [
            self.nb.dynamic_wander(self._encode(state, a))[0] for a in range(n_actions)
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
        next_q = 0.0
        if not done:
            next_q = max(
                self.nb.dynamic_wander(self._encode(next_state, a))[0]
                for a in range(4)
            )
        target = reward + self.discount * next_q
        self.nb.train([(self._encode(state, action), target)], epochs=1)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_gridworld(agent: MarbleQLearningAgent, env: GridWorld, episodes: int, max_steps: int = 50) -> list[float]:
    rewards = []
    for _ in range(episodes):
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
