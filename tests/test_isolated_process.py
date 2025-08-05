import os

from pipeline import Pipeline


def test_isolated_step_runs_in_child_process():
    parent = os.getpid()
    steps = [{"func": "getpid", "module": "os", "isolated": True}]
    pipe = Pipeline(steps)
    result = pipe.execute()[0]
    assert result != parent
