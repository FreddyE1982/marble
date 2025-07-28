import importlib

import global_workspace


def test_global_workspace_broadcast():
    importlib.reload(global_workspace)
    gw = global_workspace.activate(capacity=2)
    messages = []
    gw.subscribe(lambda m: messages.append((m.source, m.content)))
    gw.publish("module1", "foo")
    gw.publish("module2", "bar")
    assert messages == [
        ("module1", "foo"),
        ("module2", "bar"),
    ]
    assert list(m.source for m in gw.queue) == ["module1", "module2"]
    gw.publish("module3", "baz")
    assert list(m.source for m in gw.queue) == ["module2", "module3"]


def test_activation_attaches_to_nb():
    importlib.reload(global_workspace)

    class DummyNB:
        pass

    nb = DummyNB()
    gw = global_workspace.activate(nb, capacity=1)
    assert hasattr(nb, "global_workspace")
    assert nb.global_workspace is gw
    assert gw.queue.maxlen == 1
