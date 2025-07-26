import os, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
import marble_brain
from tests.test_core_functions import minimal_params


def test_usage_profiler_records(tmp_path):
    random.seed(0)
    marble_brain.tqdm = __import__("tqdm").tqdm
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    log_path = tmp_path / "prof.csv"
    brain = marble_brain.Brain(
        core,
        nb,
        DataLoader(),
        profile_enabled=True,
        profile_log_path=str(log_path),
        profile_interval=1,
    )
    examples = [(0.1, 0.1)]
    brain.train(examples, epochs=2)
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 2
