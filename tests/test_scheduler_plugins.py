from scheduler_plugins import configure_scheduler


def test_thread_scheduler_executes_tasks():
    sched = configure_scheduler("thread")
    fut = sched.schedule(lambda x: x + 1, 1)
    assert fut.result() == 2


def test_asyncio_scheduler_executes_coroutines():
    async def coro(x):
        return x * 2

    sched = configure_scheduler("asyncio")
    fut = sched.schedule(coro, 3)
    assert fut.result() == 6

    # Switch back to thread scheduler to close event loop
    configure_scheduler("thread")
