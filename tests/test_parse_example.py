import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marble_brain import _parse_example


def test_parse_example_text():
    inp, tgt = _parse_example({"input": "hello", "target": "world"})
    assert isinstance(inp, float)
    assert isinstance(tgt, float)
