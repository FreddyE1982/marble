import torch


def branch_a(device: str = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return f"a_{dev}"


def branch_b(device: str = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return f"b_{dev}"


def merge_branches(branches):
    return ",".join(branches)


def failing_step():
    raise RuntimeError("branch failure")
