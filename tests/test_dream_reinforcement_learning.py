import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch

from dream_reinforcement_learning import DreamReinforcementLearner
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_dream_reinforcement_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = DreamReinforcementLearner(core, nb, dream_cycles=1)
    err = learner.train_episode(0.5, 1.0)
    assert isinstance(err, float)
    assert learner.history


def test_dream_interval_and_duration():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learner = DreamReinforcementLearner(
        core, nb, dream_cycles=1, dream_interval=2, dream_cycle_duration=0.001
    )
    with patch.object(
        DreamReinforcementLearner, "_dream_step", wraps=learner._dream_step
    ) as mock_dream:
        learner.train_episode(0.2, 0.4)
        assert mock_dream.call_count == 0
        learner.train_episode(0.3, 0.6)
        assert mock_dream.call_count == 1
    # verify duration triggers sleep
    with patch("time.sleep") as mock_sleep:
        learner._dream_step(0.1)
        mock_sleep.assert_called_once_with(0.001)
