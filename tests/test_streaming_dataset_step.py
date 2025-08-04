import asyncio

import torch

from bit_tensor_dataset import BitTensorDataset
from streaming_dataset_step import StreamingDatasetStep


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
