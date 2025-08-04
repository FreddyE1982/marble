from __future__ import annotations

import jsonschema
import pytest

from pipeline import Pipeline


def add_one(x: int) -> int:
    return x + 1


def test_step_validation_and_execution() -> None:
    pipe = Pipeline()
    pipe.add_step(func="add_one", module=__name__, params={"x": 1})
    result = pipe.execute()[0]
    assert result == 2


def test_invalid_step_rejected() -> None:
    pipe = Pipeline(steps=[{"name": "broken"}])
    with pytest.raises(jsonschema.ValidationError):
        pipe.execute()
