from pipeline import Pipeline


def test_execute_reorders_steps():
    store = []
    steps = [
        {
            "name": "second",
            "func": "append_value",
            "module": "tests.helpers",
            "params": {"value": "b", "store": store},
            "depends_on": ["first"],
        },
        {
            "name": "first",
            "func": "append_value",
            "module": "tests.helpers",
            "params": {"value": "a", "store": store},
        },
    ]
    pipe = Pipeline(steps)
    pipe.execute()
    assert store == ["a", "b"]
