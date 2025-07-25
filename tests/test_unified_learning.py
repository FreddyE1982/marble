import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from hebbian_learning import HebbianLearner
from autoencoder_learning import AutoencoderLearner
from unified_learning import UnifiedLearner


def test_unified_learner_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learners = {
        "hebbian": HebbianLearner(core, nb),
        "auto": AutoencoderLearner(core, nb),
    }
    learner = UnifiedLearner(core, nb, learners)
    learner.train_step((0.1, 0.2))
    assert learner.history
    assert set(learner.history[0]["weights"].keys()) == set(learners.keys())


def test_unified_learner_explain_gradients():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    learners = {
        "hebbian": HebbianLearner(core, nb),
        "auto": AutoencoderLearner(core, nb),
    }
    learner = UnifiedLearner(core, nb, learners)
    learner.train_step((0.1, 0.2))
    result = learner.explain(0, with_gradients=True)
    assert "gradients" in result
    assert set(result["gradients"].keys()) == set(learners.keys())
    for g in result["gradients"].values():
        assert len(g) == 4
