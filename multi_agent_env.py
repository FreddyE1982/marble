"""Simple cooperative and competitive multi-agent environments."""

from __future__ import annotations

from typing import Dict, Tuple


class BaseMultiAgentEnv:
    """Base class for toy multi-agent reinforcement learning environments."""

    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids
        self.step_count = 0

    def reset(self) -> Dict[str, int]:
        """Reset environment state and return initial observations."""
        self.step_count = 0
        return {aid: 0 for aid in self.agent_ids}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, float], bool, dict]:
        raise NotImplementedError


class CooperativeEnv(BaseMultiAgentEnv):
    """Reward agents when they select the same action."""

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, float], bool, dict]:
        self.step_count += 1
        reward = 1.0 if len(set(actions.values())) == 1 else 0.0
        rewards = {aid: reward for aid in self.agent_ids}
        done = self.step_count >= 10
        obs = {aid: self.step_count for aid in self.agent_ids}
        return obs, rewards, done, {}


class CompetitiveEnv(BaseMultiAgentEnv):
    """Zero-sum game where highest action gains reward."""

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, float], bool, dict]:
        self.step_count += 1
        max_action = max(actions.values())
        winners = [aid for aid, act in actions.items() if act == max_action]
        rewards = {aid: (1.0 if aid in winners else -1.0) for aid in self.agent_ids}
        done = self.step_count >= 10
        obs = {aid: self.step_count for aid in self.agent_ids}
        return obs, rewards, done, {}


def run_episode(env: BaseMultiAgentEnv, agents: Dict[str, object], *, steps: int = 10) -> Dict[str, float]:
    """Run ``steps`` iterations of ``env`` using ``agents``.

    Agents must implement ``act(observation)`` and ``on_reward(reward)`` methods.
    Returns accumulated rewards per agent.
    """

    observations = env.reset()
    totals = {aid: 0.0 for aid in env.agent_ids}
    for _ in range(steps):
        acts = {aid: agents[aid].act(observations[aid]) for aid in env.agent_ids}
        observations, rewards, done, _ = env.step(acts)
        for aid, r in rewards.items():
            totals[aid] += r
            agents[aid].on_reward(r)
        if done:
            break
        # Provide observation to agents via messages if they support it.
        for aid, agent in agents.items():
            if hasattr(agent, "bus") and agent.bus:
                agent.bus.broadcast(aid, {"observation": observations[aid]})
    return totals

