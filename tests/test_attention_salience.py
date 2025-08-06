import importlib

import yaml

import attention_codelets
import global_workspace
from config_loader import create_marble_from_config, load_config


def test_attention_salience_adjustment():
    importlib.reload(attention_codelets)

    def codelet_low():
        return attention_codelets.AttentionProposal(score=0.1, content="low")

    def codelet_high():
        return attention_codelets.AttentionProposal(score=0.2, content="high")

    attention_codelets.register_codelet(codelet_low)
    attention_codelets.register_codelet(codelet_high)

    gw = global_workspace.activate(capacity=2)
    attention_codelets.activate(coalition_size=1, salience_weight=2.0)
    coalition = attention_codelets.form_coalition(saliences=[0.9, 0.1])
    assert coalition[0].content == "low"
    attention_codelets.broadcast_coalition(coalition)
    assert gw.queue[-1].content["content"] == "low"


def test_salience_weight_from_config(tmp_path):
    importlib.reload(attention_codelets)
    cfg = load_config()
    cfg["attention_codelets"] = {"enabled": True, "coalition_size": 1}
    cfg["core"]["salience_weight"] = 2.5
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    create_marble_from_config(str(cfg_path))
    assert attention_codelets._salience_weight == 2.5
