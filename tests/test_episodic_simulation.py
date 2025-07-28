import importlib

import episodic_memory
import episodic_simulation
from marble_neuronenblitz import Neuronenblitz
from marble_core import Core


def test_simulate_returns_rewards():
    importlib.reload(episodic_memory)
    importlib.reload(episodic_simulation)
    core = Core(width=1, height=1)
    nb = Neuronenblitz(core, max_wander_depth=1)
    mem = episodic_memory.EpisodicMemory(transient_capacity=3)
    mem.add_episode({"state": 1}, reward=1.0, outcome=None)
    mem.add_episode({"state": 2}, reward=2.0, outcome=None)
    rewards = episodic_simulation.simulate(nb, mem, length=2)
    assert len(rewards) == 2
