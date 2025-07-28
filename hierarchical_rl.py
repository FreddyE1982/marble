"""Hierarchical reinforcement learning utilities."""

from __future__ import annotations

from typing import Iterable

from reinforcement_learning import RLEnvironment, MarbleQLearningAgent


class LowLevelPolicy:
    """Simple wrapper around :class:`MarbleQLearningAgent`."""

    def __init__(self, env: RLEnvironment, agent: MarbleQLearningAgent) -> None:
        self.env = env
        self.agent = agent

    def select_action(self, state) -> int:
        return self.agent.select_action(state)


class HighLevelController:
    """Execute a sequence of low-level policies to achieve a goal."""

    def __init__(self, env: RLEnvironment, policies: Iterable[LowLevelPolicy]):
        self.env = env
        self.policies = list(policies)

    def act(self, state):
        """Run all policies in order until ``done`` is reached."""
        reward_total = 0.0
        for policy in self.policies:
            action = policy.select_action(state)
            state, reward, done = self.env.step(action)
            reward_total += reward
            if done:
                break
        return state, reward_total, done
