from marble_neuronenblitz import Neuronenblitz
from marble_neuronenblitz.core import Neuronenblitz as CoreNB


def test_wrapper_import():
    assert Neuronenblitz is CoreNB
