from async_utils import async_transform


def test_async_transform():
    data = [1, 2, 3]

    def fn(x):
        return x * x

    out = async_transform(data, fn)
    assert out == [1, 4, 9]
