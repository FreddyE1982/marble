import asyncio
import time

import pytest
import torch

from bit_tensor_dataset import BitTensorDataset
from streaming_dataset_step import StreamingDatasetStep
from pipeline import Pipeline
import marble_interface


def _consume(step: StreamingDatasetStep):
    async def _run():
        batches = []
        while True:
            batch = await step.next_batch()
            if batch is None:
                break
            batches.append(batch)
        return batches

    return asyncio.run(_run())


def test_streaming_dataset_step_cpu():
    ds = BitTensorDataset([(i, i) for i in range(5)], device="cpu")
    step = StreamingDatasetStep(ds, batch_size=2, prefetch=2, device="cpu")
    batches = _consume(step)
    inputs = torch.cat([b["inputs"] for b in batches])
    targets = torch.cat([b["targets"] for b in batches])
    decoded_inp = [ds.tensor_to_object(t) for t in inputs]
    decoded_tgt = [ds.tensor_to_object(t) for t in targets]
    assert decoded_inp == list(range(5))
    assert decoded_tgt == list(range(5))
    assert step.is_finished()


def test_streaming_dataset_step_variable_rates():
    ds = BitTensorDataset([(i, i) for i in range(20)], device="cpu")

    class SlowDataset:
        def __init__(self, base):
            self.base = base

        def __iter__(self):
            for idx, pair in enumerate(iter(self.base)):
                time.sleep(0.001 * (idx % 3))
                yield pair

    step = StreamingDatasetStep(SlowDataset(ds), batch_size=3, prefetch=2, device="cpu")

    async def _run():
        batches = []
        while True:
            batch = await step.next_batch()
            if batch is None:
                break
            await asyncio.sleep(0.001 * (len(batches) % 2))
            batches.append(batch)
        return batches

    batches = asyncio.run(_run())
    inputs = torch.cat([b["inputs"] for b in batches])
    decoded = [ds.tensor_to_object(t) for t in inputs]
    assert decoded == list(range(20))
    assert step.is_finished()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_streaming_dataset_step_gpu():
    ds = BitTensorDataset([(i, i) for i in range(5)], device="cpu")
    cpu_step = StreamingDatasetStep(ds, batch_size=2, prefetch=2, device="cpu")
    gpu_step = StreamingDatasetStep(ds, batch_size=2, prefetch=2, device="cuda")
    cpu_batches = _consume(cpu_step)
    gpu_batches = _consume(gpu_step)
    cpu_inputs = torch.cat([b["inputs"] for b in cpu_batches])
    gpu_inputs = torch.cat([b["inputs"] for b in gpu_batches]).to("cpu")
    assert torch.equal(cpu_inputs, gpu_inputs)
    decoded = [ds.tensor_to_object(t) for t in gpu_inputs]
    assert decoded == list(range(5))


def test_pipeline_auto_consumes_streaming_step():
    ds = BitTensorDataset([(i, i) for i in range(4)], device="cpu")
    pipe = Pipeline([
        {
            "func": "streaming_dataset_step",
            "module": "marble_interface",
            "params": {"dataset": ds, "batch_size": 2, "prefetch": 2, "device": "cpu"},
        }
    ])
    result = pipe.execute()
    batches = result[0]
    inputs = torch.cat([b["inputs"] for b in batches])
    decoded = [ds.tensor_to_object(t) for t in inputs]
    assert decoded == list(range(4))
