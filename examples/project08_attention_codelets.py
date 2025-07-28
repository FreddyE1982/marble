"""Demonstrate attention codelet coalition formation."""

import time
from global_workspace import activate as gw_activate
import attention_codelets as ac


def codelet_a():
    return ac.AttentionProposal(score=0.2, content="hello")


def codelet_b():
    return ac.AttentionProposal(score=0.8, content="world")


def main():
    gw_activate(capacity=5)
    ac.register_codelet(codelet_a)
    ac.register_codelet(codelet_b)
    ac.activate(coalition_size=2)
    for _ in range(3):
        ac.run_cycle()
        time.sleep(0.1)
    for msg in ac.global_workspace.workspace.queue:
        print(msg.content)


if __name__ == "__main__":
    main()
