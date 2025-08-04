import pytest
import torch

from pipeline import Pipeline


def test_pipeline_executes_in_dependency_order_cpu():
    p = Pipeline()
    p.add_step("step_c", module="tests.dependency_steps", name="c", depends_on=["b"])
    p.add_step("step_a", module="tests.dependency_steps", name="a")
    p.add_step("step_b", module="tests.dependency_steps", name="b", depends_on=["a"])
    results = p.execute()
    names = [r[0] for r in results]
    devices = [r[1] for r in results]
    assert names == ["a", "b", "c"]
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert all(d == expected_device for d in devices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pipeline_executes_on_gpu():
    p = Pipeline()
    p.add_step("step_a", module="tests.dependency_steps", name="a")
    p.add_step("step_b", module="tests.dependency_steps", name="b", depends_on=["a"])
    results = p.execute()
    devices = [r[1] for r in results]
    assert all(d == "cuda" for d in devices)


def test_cycle_detection_in_execute():
    p = Pipeline()
    p.add_step("step_a", module="tests.dependency_steps", name="a", depends_on=["c"])
    p.add_step("step_b", module="tests.dependency_steps", name="b", depends_on=["a"])
    p.add_step("step_c", module="tests.dependency_steps", name="c", depends_on=["b"])
    with pytest.raises(ValueError, match="cycle"):
        p.execute()
