import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble_core import Core
from tests.test_core_functions import minimal_params


def test_choose_new_tier_prefers_ram_when_under_limit():
    params = minimal_params()
    params['vram_limit_mb'] = 0
    params['ram_limit_mb'] = 0.001
    core = Core(params)
    assert core.choose_new_tier() == 'ram'


def test_choose_new_tier_uses_disk_when_ram_full():
    params = minimal_params()
    params['vram_limit_mb'] = 0
    params['ram_limit_mb'] = 0.0001
    params['tier_autotune_enabled'] = False
    core = Core(params)
    assert core.choose_new_tier() == 'disk'


def test_autotune_tiers_migrates_neurons():
    params = minimal_params()
    params['vram_limit_mb'] = 0.0001
    params['ram_limit_mb'] = 0.0002
    params['tier_autotune_enabled'] = True
    core = Core(params)
    vram_neurons = sum(n.tier == 'vram' for n in core.neurons)
    has_ram = any(n.tier == 'ram' for n in core.neurons)
    has_disk = any(n.tier == 'disk' for n in core.neurons)
    assert vram_neurons == 0
    assert has_ram or has_disk
