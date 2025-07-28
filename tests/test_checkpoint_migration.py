import pickle
import random
import numpy as np

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from tests.test_core_functions import minimal_params


def test_checkpoint_migration(tmp_path):
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    nb.context_history.append({"arousal": 0.1, "stress": 0.2, "reward": 0.3})
    nb.replay_buffer.append((1.0, 0.0))
    state = {
        "epoch": 1,
        "core": core,
        "neuronenblitz": nb,
        "meta_controller": None,
        "memory_system": None,
        "lobe_manager": None,
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
    }
    ckpt = tmp_path / "old.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(state, f)

    brain = Brain(core, nb, None, save_dir=str(tmp_path))
    epoch = brain.load_checkpoint(str(ckpt))
    ctx = brain.neuronenblitz.context_history[-1]
    assert epoch == 1
    assert "markers" in ctx and "goals" in ctx and "tom" in ctx
    assert len(brain.neuronenblitz.replay_buffer[0]) == 5
