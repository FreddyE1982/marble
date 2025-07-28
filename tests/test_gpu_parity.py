import numpy as np
import torch
import marble_core


def test_simple_mlp_cpu_gpu_parity():
    if not torch.cuda.is_available():
        assert True
        return
    x_np = np.random.randn(2, marble_core._REP_SIZE).astype(np.float32)
    cpu_out = marble_core._simple_mlp(x_np)
    x_gpu = torch.tensor(x_np, device="cuda")
    gpu_out = marble_core._simple_mlp(x_gpu).cpu().numpy()
    assert np.allclose(cpu_out, gpu_out, atol=1e-6)
