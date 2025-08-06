import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from highlevel_pipeline import HighLevelPipeline


def test_default_memory_limit_enforced(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("pipeline:\n  default_step_memory_limit_mb: 1\n")
    pipe = HighLevelPipeline(config_path=str(cfg_path))

    def allocate():
        return bytearray(2 * 1024 * 1024)

    pipe.add_step(allocate)
    with pytest.raises(MemoryError):
        pipe.execute()
