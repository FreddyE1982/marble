import torch


def test_torch_version_compatible():
    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    assert version >= (2, 7)
