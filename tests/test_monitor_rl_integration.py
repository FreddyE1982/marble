from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from reinforcement_learning import MarbleQLearningAgent, GridWorld
import self_monitoring
from tests.test_core_functions import minimal_params


def test_monitor_adjusts_parameters():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, monitor_wander_factor=0.1, monitor_epsilon_factor=0.2)
    self_monitoring.activate(nb)
    agent = MarbleQLearningAgent(core, nb)
    self_monitoring.log_error(0.5)
    prev_noise = nb.wander_depth_noise
    prev_eps = agent.epsilon
    agent.select_action((0, 0), 4)
    assert nb.wander_depth_noise > prev_noise
    assert agent.epsilon <= prev_eps
