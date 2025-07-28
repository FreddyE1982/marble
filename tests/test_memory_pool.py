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
