import torch

import attention_codelets as ac
import global_workspace


def test_workspace_events_affect_gating():
    global_workspace.activate()
    ac._codelets.clear()
    ac._workspace_gates.clear()

    def c1():
        return ac.AttentionProposal(score=0.1, content="a")

    def c2():
        return ac.AttentionProposal(score=0.2, content="b")

    ac.register_codelet(c1)
    ac.register_codelet(c2)
    ac.enable_workspace_gating()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    winner = ac.form_coalition(coalition_size=1, device=device)[0]
    assert winner.content == "b"
    global_workspace.workspace.publish("test", {"codelet": "c1", "gate": 1.0})
    winner = ac.form_coalition(coalition_size=1, device=device)[0]
    assert winner.content == "a"
