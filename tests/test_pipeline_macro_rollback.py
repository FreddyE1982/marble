import torch
from pathlib import Path
from pipeline import Pipeline


def test_macro_step_execution(tmp_path: Path):
    p = Pipeline()
    p.add_step("step_a", module="tests.dependency_steps", name="a")
    macro_steps = [
        {"func": "step_b", "module": "tests.dependency_steps", "name": "b"},
        {"func": "step_c", "module": "tests.dependency_steps", "name": "c"},
    ]
    p.add_macro("macro_bc", macro_steps)
    results = p.execute(cache_dir=tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert results[0][0] == "a"
    assert results[0][1] == device
    assert [r[0] for r in results[1]] == ["b", "c"]
    assert all(r[1] == device for r in results[1])


def test_rollback_removes_future_cache(tmp_path: Path):
    p = Pipeline()
    p.add_step("step_a", module="tests.dependency_steps", name="a")
    p.add_step("step_b", module="tests.dependency_steps", name="b")
    p.add_step("step_c", module="tests.dependency_steps", name="c")
    results = p.execute(cache_dir=tmp_path)
    assert len(results) == 3
    # Make step_b fail
    p.steps[1]["func"] = "failing_step"
    p.steps[1]["module"] = "tests.branching_steps"
    try:
        p.execute(cache_dir=tmp_path)
    except RuntimeError:
        pass
    rolled = p.rollback("a", tmp_path)
    assert rolled[0] == "a"
    assert any(tmp_path.glob("0_a_*.pt"))
    assert not any(tmp_path.glob("1_b_*.pt"))
    assert not any(tmp_path.glob("2_c_*.pt"))
    # restore
    p.steps[1]["func"] = "step_b"
    p.steps[1]["module"] = "tests.dependency_steps"
    results2 = p.execute(cache_dir=tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert results2[0][1] == device
    assert len(results2) == 3
