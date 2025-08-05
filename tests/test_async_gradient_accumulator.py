import asyncio

import pytest
import torch

from async_gradient_accumulator import AsyncGradientAccumulator


def _sync_train(model, optim, loss_fn, data, accumulation_steps, device):
    model.to(device)
    optim.zero_grad(set_to_none=True)
    counter = 0
    for inp, tgt in data:
        inp = inp.to(device)
        tgt = tgt.to(device)
        out = model(inp)
        loss = loss_fn(out, tgt)
        loss.backward()
        counter += 1
        if counter >= accumulation_steps:
            optim.step()
            optim.zero_grad(set_to_none=True)
            counter = 0
    if counter > 0:
        optim.step()
        optim.zero_grad(set_to_none=True)


def _make_data(device):
    xs = torch.randn(8, 4, device=device)
    ys = torch.randn(8, 1, device=device)
    return list(zip(xs, ys))


def test_async_gradient_accumulator_cpu():
    device = torch.device("cpu")
    data = _make_data(device)
    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(4, 1)
    model_b.load_state_dict(model_a.state_dict())
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    _sync_train(model_a, opt_a, loss_fn, data, 4, device)

    async def _run():
        acc = AsyncGradientAccumulator(
            model_b, opt_b, loss_fn, accumulation_steps=4, device=device
        )
        for inp, tgt in data:
            await acc.add_batch(inp, tgt)
        await acc.flush()

    asyncio.run(_run())
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p1, p2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_async_gradient_accumulator_gpu():
    device = torch.device("cuda")
    data = _make_data(device)
    model_a = torch.nn.Linear(4, 1).to(device)
    model_b = torch.nn.Linear(4, 1).to(device)
    model_b.load_state_dict(model_a.state_dict())
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()

    _sync_train(model_a, opt_a, loss_fn, data, 4, device)

    async def _run():
        acc = AsyncGradientAccumulator(
            model_b, opt_b, loss_fn, accumulation_steps=4, device=device
        )
        for inp, tgt in data:
            await acc.add_batch(inp, tgt)
        await acc.flush()

    asyncio.run(_run())
    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p1, p2)
