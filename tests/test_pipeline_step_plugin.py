import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import Pipeline
from pipeline_plugins import PLUGIN_REGISTRY, load_pipeline_plugins


def test_registry_and_lookup(tmp_path):
    plugin_code = (
        "from pipeline_plugins import PipelinePlugin, register_plugin\n"
        "import torch\n"
        "class TmpPlugin(PipelinePlugin):\n"
        "    def initialise(self, device, marble=None):\n        \tself.device = device\n"
        "    def execute(self, device, marble=None):\n        \treturn torch.ones(1, device=device)\n"
        "    def teardown(self):\n        \tpass\n"
        "def register(reg):\n    \treg('tmp', TmpPlugin)\n"
    )
    plugin_file = tmp_path / "tmp_plugin.py"
    plugin_file.write_text(plugin_code)
    load_pipeline_plugins([str(tmp_path)])
    assert "tmp" in PLUGIN_REGISTRY


def test_pipeline_plugin_execution(tmp_path):
    plugin_code = (
        "from pipeline_plugins import PipelinePlugin, register_plugin\n"
        "import torch\n"
        "class P(PipelinePlugin):\n"
        "    def __init__(self, factor=2):\n        \tself.factor = factor\n"
        "    def initialise(self, device, marble=None):\n        \tself.device = device\n"
        "    def execute(self, device, marble=None):\n        \treturn torch.ones(1, device=device) * self.factor\n"
        "    def teardown(self):\n        \tpass\n"
        "def register(reg):\n    \treg('mult', P)\n"
    )
    plugin_file = tmp_path / "mult_plugin.py"
    plugin_file.write_text(plugin_code)
    load_pipeline_plugins([str(tmp_path)])

    pipe = Pipeline([{ "plugin": "mult", "params": {"factor": 3}}])
    result = pipe.execute()[0]
    assert torch.allclose(result, torch.tensor([3.], device=result.device))
    assert result.device.type in {"cpu", "cuda"}
