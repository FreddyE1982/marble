import asyncio
from highlevel_pipeline import HighLevelPipeline

async def async_step(x):
    await asyncio.sleep(0.01)
    return x * 2


def sync_step(y):
    return y + 1


def test_execute_async():
    hp = HighLevelPipeline()
    hp.add_step(async_step, params={"x": 3})
    hp.add_step(sync_step, params={"y": 5})
    marble, results = asyncio.run(hp.execute_async())
    assert results == [6, 6]
