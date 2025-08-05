import torch

from learning_plugins import get_learning_module, load_learning_plugins
from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
from unified_learning import UnifiedLearner


def test_learning_module_plugin_swapping(tmp_path):
    plugin1 = (
        "import torch\n"
        "from learning_plugins import LearningModule, register_learning_module\n"
        "class AddLearner(LearningModule):\n"
        "    def initialise(self, device, marble=None):\n"
        "        self.device = device\n"
        "    def train_step(self, x, y, *, device, marble=None):\n"
        "        return torch.tensor(x, device=device) + torch.tensor(y, device=device)\n"
        "def register(reg):\n"
        "    reg('math', AddLearner)\n"
    )
    file = tmp_path / "plugin.py"
    file.write_text(plugin1)
    load_learning_plugins(str(tmp_path))
    cls = get_learning_module("math")
    learner = cls()
    learner.initialise(torch.device("cpu"))
    res = learner.train_step(1.0, 2.0, device=torch.device("cpu"))
    assert res.item() == 3.0
    file.unlink()

    plugin2 = (
        "import torch\n"
        "from learning_plugins import LearningModule, register_learning_module\n"
        "class MulLearner(LearningModule):\n"
        "    def initialise(self, device, marble=None):\n"
        "        self.device = device\n"
        "    def train_step(self, x, y, *, device, marble=None):\n"
        "        return torch.tensor(x, device=device) * torch.tensor(y, device=device)\n"
        "def register(reg):\n"
        "    reg('math', MulLearner)\n"
    )
    file2 = tmp_path / "plugin2.py"
    file2.write_text(plugin2)
    load_learning_plugins(str(tmp_path))
    cls2 = get_learning_module("math")
    learner2 = cls2()
    learner2.initialise(torch.device("cpu"))
    res2 = learner2.train_step(2.0, 3.0, device=torch.device("cpu"))
    assert res2.item() == 6.0


def test_unified_learner_uses_plugin(tmp_path):
    plugin = (
        "import torch\n"
        "from learning_plugins import LearningModule, register_learning_module\n"
        "class LossLearner(LearningModule):\n"
        "    def initialise(self, device, marble=None):\n"
        "        self.device = device\n"
        "    def train_step(self, x, y, *, device, marble=None):\n"
        "        pred = torch.tensor(x, device=device)\n"
        "        target = torch.tensor(y, device=device)\n"
        "        return (pred - target).abs().sum()\n"
        "def register(reg):\n"
        "    reg('loss', LossLearner)\n"
    )
    file = tmp_path / "loss_plugin.py"
    file.write_text(plugin)
    core = Core(minimal_params())
    nb = Neuronenblitz(core)
    ul = UnifiedLearner(core, nb, {"l": "loss"}, plugin_dirs=[str(tmp_path)])
    ul.train_step((2.0, 1.0))
    assert ul.loss_history["l"]
