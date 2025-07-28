"""Example training script using hierarchical RL."""

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from reinforcement_learning import GridWorld, MarbleQLearningAgent
from hierarchical_rl import LowLevelPolicy, HighLevelController


if __name__ == "__main__":
    params = {
        "xmin": -2.0,
        "xmax": 1.0,
        "ymin": -1.5,
        "ymax": 1.5,
        "width": 4,
        "height": 4,
    }
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    env = GridWorld(size=4)
    agent = MarbleQLearningAgent(core, nb)
    policy = LowLevelPolicy(env, agent)
    controller = HighLevelController(env, [policy])
    state = env.reset()
    for _ in range(20):
        state, reward, done = controller.act(state)
        if done:
            break
    print("Final state", state, "reward", reward)
