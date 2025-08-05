import pytest
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


def test_add_to_replay_stores_neuromod_signals():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core, use_experience_replay=True)
    nb.update_context(arousal=0.2, stress=0.3, reward=0.4, emotion=0.6)
    nb.add_to_replay(0.1, 0.2, 0.05)
    stored = nb.replay_buffer[-1]
    assert stored[-4] == pytest.approx(0.2)
    assert stored[-3] == pytest.approx(0.3)
    assert stored[-2] == pytest.approx(0.4)
    assert stored[-1] == pytest.approx(0.6)
