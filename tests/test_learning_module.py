from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.learning import enable_rl, rl_select_action, rl_update
from tests.test_core_functions import minimal_params


def test_rl_functions_work():
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    enable_rl(nb)
    action = rl_select_action(nb, (0, 0), 2)
    rl_update(nb, (0, 0), action, 1.0, (0, 1), False, n_actions=2)
    assert nb.rl_enabled
