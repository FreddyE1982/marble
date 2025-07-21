import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from reinforcement_learning import GridWorld, MarbleQLearningAgent, train_gridworld
from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz


def main() -> None:
    ds = load_dataset("mnist", split="train[:10]")
    size = 4 + len(ds) % 3
    env = GridWorld(size=size)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    agent = MarbleQLearningAgent(core, nb)
    rewards = train_gridworld(agent, env, episodes=2, max_steps=20)
    print("Rewards:", rewards)


if __name__ == "__main__":
    main()
