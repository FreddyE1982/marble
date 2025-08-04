import pytest

from pipeline import Pipeline


def test_topological_sort_orders_steps():
    p = Pipeline()
    steps = [
        {"name": "c", "depends_on": ["b"]},
        {"name": "a"},
        {"name": "b", "depends_on": ["a"]},
    ]
    ordered = p._topological_sort(steps)
    assert [s["name"] for s in ordered] == ["a", "b", "c"]


def test_cycle_detection_raises_error():
    p = Pipeline()
    steps = [
        {"name": "a", "depends_on": ["b"]},
        {"name": "b", "depends_on": ["a"]},
    ]
    with pytest.raises(ValueError, match="cycle"):
        p._topological_sort(steps)


def test_unknown_dependency_error():
    p = Pipeline()
    steps = [{"name": "a", "depends_on": ["missing"]}]
    with pytest.raises(ValueError, match="unknown step"):
        p._topological_sort(steps)
