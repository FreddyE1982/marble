import torch

def scale_value(a: float, scale: float = 1.0, device: str = "cpu") -> float:
    tensor = torch.tensor(a, device=device, dtype=torch.float32)
    return float(tensor * scale)
