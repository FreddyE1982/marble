import time
import torch

import time
import torch

from tensor_sync_service import compute_delta, apply_delta, TensorSyncService


def test_compute_and_apply_delta_int():
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([0, 2, 1], dtype=torch.int32)
    delta = compute_delta(a, b)
    assert torch.equal(delta, torch.tensor([1, 0, 2], dtype=torch.int32))
    apply_delta(b, delta)
    assert torch.equal(a, b)


def test_compute_and_apply_delta_float():
    a = torch.tensor([1.5, 2.5])
    b = torch.tensor([0.5, 1.0])
    delta = compute_delta(a, b)
    assert torch.allclose(delta, torch.tensor([1.0, 1.5]))
    apply_delta(b, delta)
    assert torch.allclose(a, b)


def _run_naive(t1, t2, updates):
    start = time.perf_counter()
    bytes_sent = 0
    for _ in range(updates):
        t1[:1000].add_(1)
        t2.copy_(t1)
        time.sleep(0.01)  # simulate network latency
        bytes_sent += t1.element_size() * t1.nelement()
    return time.perf_counter() - start, bytes_sent


def _run_service(t1, t2, updates):
    svc = TensorSyncService(interval_ms=5)
    svc.register("a", t1)
    svc.register("b", t2)
    start = time.perf_counter()
    for _ in range(updates):
        t1[:1000].add_(1)
        time.sleep(0.005)  # allow background sync
    svc.stop()
    return time.perf_counter() - start, svc.metrics["bytes_sent"]


def test_tensor_sync_service_reduces_latency_and_volume():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = torch.zeros(100000, dtype=torch.float32, device=device)
    t2 = torch.zeros_like(t1)
    naive_time, naive_bytes = _run_naive(t1.clone(), t2.clone(), 3)
    svc_time, svc_bytes = _run_service(t1, t2, 3)
    assert svc_bytes < naive_bytes
    assert svc_time < naive_time
