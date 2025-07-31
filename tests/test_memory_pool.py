from memory_pool import MemoryPool

class Dummy:
    def __init__(self):
        self.value = 0

def test_memory_pool_allocate_release():
    pool = MemoryPool(Dummy, max_size=2)
    a = pool.allocate()
    pool.release(a)
    b = pool.allocate()
    assert a is b
    pool.release(b)


def test_memory_pool_preallocate_and_borrow():
    pool = MemoryPool(Dummy, max_size=3)
    pool.preallocate(2)
    assert len(pool) == 2
    with pool.borrow() as obj:
        assert isinstance(obj, Dummy)
        obj.value = 42
    # object should have been released back to pool
    assert len(pool) == 2
