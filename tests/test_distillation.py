import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_core_functions import minimal_params
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from distillation_trainer import DistillationTrainer


def build_brain():
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    dl = DataLoader()
    return Brain(core, nb, dl)


def simple_dataset():
    return [(0.1, 0.2), (0.2, 0.4), (0.3, 0.6)]


def test_distillation_reduces_student_error():
    teacher = build_brain()
    teacher.train(simple_dataset(), epochs=3)

    student = build_brain()
    trainer = DistillationTrainer(student, teacher, alpha=0.5)

    inp = 0.25
    teacher_pred = teacher.infer(inp)
    student_before = student.infer(inp)

    trainer.train(simple_dataset(), epochs=2)

    student_after = student.infer(inp)
    err_before = abs(student_before - teacher_pred)
    err_after = abs(student_after - teacher_pred)
    assert err_after <= err_before
