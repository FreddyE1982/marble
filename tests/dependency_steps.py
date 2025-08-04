import torch


def step_a(device: str = "cpu"):
    return ("a", device, torch.tensor(1, device=device))


def step_b(device: str = "cpu"):
    return ("b", device, torch.tensor(2, device=device))


def step_c(device: str = "cpu"):
    return ("c", device, torch.tensor(3, device=device))
