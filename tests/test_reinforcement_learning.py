import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch.nn as nn

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from reinforcement_learning import (
    GridWorld,
    MarblePolicyGradientAgent,
    MarbleQLearningAgent,
    train_gridworld,
    train_policy_gradient,
)
from tests.test_core_functions import minimal_params


def test_qlearning_improves_reward():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarbleQLearningAgent(core, nb, discount=0.9, epsilon=1.0, epsilon_decay=0.8)
    env = GridWorld(size=3)
    rewards = train_gridworld(agent, env, episodes=5, max_steps=20)
    assert rewards[-1] >= rewards[0] - 1e-9


def test_neuronenblitz_rl_toggle():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    with pytest.raises(RuntimeError):
        nb.rl_select_action((0, 0), 2)
    nb.enable_rl()
    action = nb.rl_select_action((0, 0), 2)
    prev_eps = nb.rl_epsilon
    nb.rl_update((0, 0), action, 1.0, (0, 1), False, n_actions=2)
    assert nb.rl_epsilon < prev_eps


def test_core_qlearning_update():
    params = minimal_params()
    params["reinforcement_learning_enabled"] = True
    core = Core(params)
    action = core.rl_select_action("s0", 2)
    core.rl_update("s0", action, 1.0, "s1", True, n_actions=2)
    assert core.q_table[("s0", action)] != 0.0


def test_double_q_learning_updates_both_tables():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarbleQLearningAgent(
        core,
        nb,
        discount=0.9,
        epsilon=0.0,
        double_q=True,
    )
    env = GridWorld(size=3)
    train_gridworld(agent, env, episodes=2, max_steps=5)
    total_entries = len(agent.q_table_a) + len(agent.q_table_b)
    assert total_entries > 0


def test_policy_gradient_improves_reward():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarblePolicyGradientAgent(core, nb, lr=0.01, gamma=0.9)
    env = GridWorld(size=3)
    rewards = train_policy_gradient(agent, env, episodes=50, max_steps=20, seed=0)
    assert rewards[-1] >= rewards[0] - 1e-9


def test_qlearning_learning_rate_parameter():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarbleQLearningAgent(core, nb, discount=0.0, epsilon=0.0, learning_rate=0.5)
    state = (0, 0)
    next_state = (0, 1)
    agent.update(state, 0, 1.0, next_state, False)
    fast_update = agent.q_table[(state, 0)]
    agent2 = MarbleQLearningAgent(
        core, nb, discount=0.0, epsilon=0.0, learning_rate=0.1
    )
    agent2.update(state, 0, 1.0, next_state, False)
    slow_update = agent2.q_table[(state, 0)]
    assert fast_update - slow_update > 0.3


def test_policy_gradient_configurable_dimensions():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarblePolicyGradientAgent(core, nb, hidden_dim=8, lr=0.02)
    assert isinstance(agent.model[0], nn.Linear)
    assert agent.model[0].out_features == 8
