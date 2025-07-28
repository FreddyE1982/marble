import importlib

import attention_codelets


def test_register_and_get_codelets():
    importlib.reload(attention_codelets)
    called = []

    def codelet(msg):
        called.append(msg)

    attention_codelets.register_codelet(codelet)
    codelets = attention_codelets.get_codelets()
    assert len(codelets) == 1
    codelets[0]("hello")
    assert called == ["hello"]
