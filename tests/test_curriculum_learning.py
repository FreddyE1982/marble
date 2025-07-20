from tests.test_core_functions import minimal_params
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from curriculum_learning import CurriculumLearner


def test_curriculum_learning_runs():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    dataset = [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0)]
    learner = CurriculumLearner(core, nb, difficulty_fn=lambda p: abs(p[0]))
    losses = learner.train(dataset, epochs=2)
    assert len(losses) > 0
    assert len(learner.history) == len(losses)
