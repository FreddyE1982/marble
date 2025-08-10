from marble_main import MARBLE
from tests.test_core_functions import minimal_params


def test_autograd_accumulation_steps_from_config():
    params = minimal_params()
    marble = MARBLE(
        params,
        dataloader_params={},
        autograd_params={
            'enabled': True,
            'learning_rate': 0.05,
            'gradient_accumulation_steps': 2,
        },
    )
    layer = marble.get_autograd_layer()
    assert layer is not None
    assert layer.accumulation_steps == 2
