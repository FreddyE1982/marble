import importlib

import attention_codelets


def test_coalition_and_broadcast():
    importlib.reload(attention_codelets)

    def codelet_a():
        return attention_codelets.AttentionProposal(score=0.2, content="a")

    def codelet_b():
        return attention_codelets.AttentionProposal(score=0.8, content="b")

    attention_codelets.register_codelet(codelet_a)
    attention_codelets.register_codelet(codelet_b)

    coalition = attention_codelets.form_coalition()
    assert len(coalition) == 1
    assert coalition[0].content == "b"

    import global_workspace

    global_workspace.activate(capacity=5)
    attention_codelets.run_cycle()
    assert global_workspace.workspace.queue[-1].content == "b"
