import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from marble_core import Core
from tests.test_core_functions import minimal_params


def test_invalid_representation_size():
    p = minimal_params()
    p["representation_size"] = 0
    with pytest.raises(ValueError):
        Core(p)


def test_invalid_message_passing_iterations():
    p = minimal_params()
    p["message_passing_iterations"] = 0
    with pytest.raises(ValueError):
        Core(p)
