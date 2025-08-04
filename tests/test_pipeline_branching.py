import pytest

from pipeline import Pipeline


def test_branch_merge_execution():
    p = Pipeline()
    p.add_branch(
        branches=[
            [{"func": "branch_a", "module": "tests.branching_steps"}],
            [{"func": "branch_b", "module": "tests.branching_steps"}],
        ],
        merge={"func": "merge_branches", "module": "tests.branching_steps"},
    )
    results = p.execute()
    assert results == ["a_cpu,b_cpu"] or results == ["a_cuda,b_cuda"]


def test_branch_failure_propagates():
    p = Pipeline()
    p.add_branch(
        branches=[
            [{"func": "branch_a", "module": "tests.branching_steps"}],
            [{"func": "failing_step", "module": "tests.branching_steps"}],
        ]
    )
    with pytest.raises(RuntimeError):
        p.execute()
