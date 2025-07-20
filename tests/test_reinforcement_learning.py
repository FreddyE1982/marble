import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reinforcement_learning import GridWorld, MarbleQLearningAgent, train_gridworld
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


def test_qlearning_improves_reward():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarbleQLearningAgent(core, nb, discount=0.9, epsilon=1.0, epsilon_decay=0.8)
    env = GridWorld(size=3)
    rewards = train_gridworld(agent, env, episodes=5, max_steps=20)
    assert rewards[-1] >= rewards[0] - 1e-9
