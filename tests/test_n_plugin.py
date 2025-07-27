import importlib
import os
import sys

from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params
import n_plugin


def test_neuronenblitz_plugin_activation(tmp_path):
    importlib.reload(n_plugin)
    plugin_code = "def activate(nb):\n    nb.plugin_value = 42\n"
    p = tmp_path / "nb_plugin.py"
    p.write_text(plugin_code)
    sys.path.insert(0, str(tmp_path))
    try:
        core = Core(minimal_params())
        nb = Neuronenblitz(core)
        n_plugin.activate("nb_plugin")
        assert getattr(nb, "plugin_value", None) == 42
    finally:
        sys.path.remove(str(tmp_path))


def test_activate_before_register(tmp_path):
    importlib.reload(n_plugin)
    plugin_code = "def activate(nb):\n    nb.plugin_value = 99\n"
    p = tmp_path / "nb_plugin2.py"
    p.write_text(plugin_code)
    sys.path.insert(0, str(tmp_path))
    try:
        n_plugin.activate("nb_plugin2")
        core = Core(minimal_params())
        nb = Neuronenblitz(core)
        assert getattr(nb, "plugin_value", None) == 99
    finally:
        sys.path.remove(str(tmp_path))
