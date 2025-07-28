import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reinforcement_learning import (
    GridWorld,
    MarblePolicyGradientAgent,
    train_policy_gradient,
)
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


def main() -> None:
    env = GridWorld(size=4)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarblePolicyGradientAgent(core, nb)
    rewards = train_policy_gradient(agent, env, episodes=10, max_steps=30)
    print("Rewards:", rewards)


if __name__ == "__main__":
    main()
