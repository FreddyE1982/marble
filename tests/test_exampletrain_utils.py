import torch

from exampletrain_utils import create_synthetic_dataset


def test_synthetic_dataset_cpu_gpu_parity():
    """Synthetic dataset should yield identical values on CPU and GPU."""

    cpu_data = create_synthetic_dataset(num_samples=5, device="cpu", seed=123)

    if torch.cuda.is_available():
        gpu_data = create_synthetic_dataset(num_samples=5, device="cuda", seed=123)

        for (cpu_in, cpu_out), (gpu_in, gpu_out) in zip(cpu_data, gpu_data):
            assert torch.allclose(cpu_in, gpu_in.cpu())
            assert torch.allclose(cpu_out, gpu_out.cpu())
    else:
        # When CUDA is unavailable ensure the CPU path still returns tensors.
        assert len(cpu_data) == 5
        first_in, first_out = cpu_data[0]
        assert isinstance(first_in, torch.Tensor)
        assert isinstance(first_out, torch.Tensor)
