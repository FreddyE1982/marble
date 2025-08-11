import random
import numpy as np
import pytest
from marble_core import Core, Neuron
from marble_neuronenblitz import Neuronenblitz
from tests.test_core_functions import minimal_params


class FailingRemote:
    def process(self, value, timeout=None):
        raise RuntimeError("remote fail")


def make_simple_core():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=0.0), Neuron(1, value=0.0)]
    core.add_synapse(0, 1, weight=1.0)
    return core


def test_exploration_decay():
    core = make_simple_core()
    nb = Neuronenblitz(core, exploration_bonus=1.0, exploration_decay=0.5,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb.dynamic_wander(1.0)
    assert nb.exploration_bonus == pytest.approx(0.5)


def test_reward_stress_scaling_and_decay():
    core = make_simple_core()
    nb = Neuronenblitz(core, reward_scale=2.0, stress_scale=3.0,
                        plasticity_modulation=1.0, reward_decay=0.5,
                        plasticity_threshold=1.0,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    nb.modulate_plasticity({"reward": 1.0, "stress": 1.0})
    assert nb.plasticity_threshold == pytest.approx(2.0)
    assert nb.last_context["reward"] == pytest.approx(1.0)
    assert nb.last_context["stress"] == pytest.approx(1.5)


def test_remote_fallback():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=1.0), Neuron(1, value=0.0, tier="remote")]
    core.add_synapse(0, 1, weight=1.0)
    nb = Neuronenblitz(core, remote_client=FailingRemote(), remote_fallback=True,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    out, _ = nb.dynamic_wander(1.0)
    assert out == pytest.approx(np.cos(0.1))
    nb_no_fb = Neuronenblitz(core, remote_client=FailingRemote(), remote_fallback=False,
                             split_probability=0.0, alternative_connection_prob=0.0,
                             backtrack_probability=0.0, backtrack_enabled=False)
    with pytest.raises(RuntimeError):
        nb_no_fb.dynamic_wander(1.0)


def test_noise_injection_std():
    core = Core(minimal_params())
    core.neurons = [Neuron(0, value=0.0)]
    nb = Neuronenblitz(core, noise_injection_std=1.0,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    np.random.seed(0)
    out, _ = nb.dynamic_wander(0.0)
    assert out != 0.0


def test_backtrack_enabled():
    params = minimal_params()
    core_bt = Core(params)
    core_bt.neurons = [Neuron(0, value=1.0), Neuron(1, value=1.0),
                       Neuron(2, value=0.0), Neuron(3, value=1.0)]
    core_bt.add_synapse(0, 1, weight=1.0)
    s_dead = core_bt.add_synapse(1, 2, weight=0.0)
    core_bt.add_synapse(1, 3, weight=1.0)
    s_dead.potential = 9.0
    np.random.seed(0)
    nb_bt = Neuronenblitz(core_bt, backtrack_probability=1.0,
                          backtrack_depth_limit=2, backtrack_enabled=True,
                          split_probability=0.0, alternative_connection_prob=0.0)
    nb_bt.dynamic_wander(1.0)
    assert s_dead.potential == 0.0
    params2 = minimal_params()
    core_no = Core(params2)
    core_no.neurons = [Neuron(0, value=1.0), Neuron(1, value=1.0),
                       Neuron(2, value=0.0), Neuron(3, value=1.0)]
    core_no.add_synapse(0, 1, weight=1.0)
    s_dead2 = core_no.add_synapse(1, 2, weight=0.0)
    core_no.add_synapse(1, 3, weight=1.0)
    s_dead2.potential = 9.0
    np.random.seed(0)
    nb_no = Neuronenblitz(core_no, backtrack_probability=1.0,
                          backtrack_depth_limit=2, backtrack_enabled=False,
                          split_probability=0.0, alternative_connection_prob=0.0)
    nb_no.dynamic_wander(1.0)
    assert s_dead2.potential > 0.0


def test_structural_plasticity_flag():
    params = minimal_params()
    core1 = Core(params)
    core1.neurons = [Neuron(0, value=1.0), Neuron(1, value=1.0)]
    syn1 = core1.add_synapse(0, 1, weight=1.0)
    syn1.potential = 100.0
    nb_off = Neuronenblitz(core1, structural_plasticity_enabled=False,
                            plasticity_threshold=1.0,
                            split_probability=0.0, alternative_connection_prob=0.0,
                            backtrack_probability=0.0, backtrack_enabled=False)
    nb_off.dynamic_wander(1.0)
    assert len(core1.neurons) == 2
    core2 = Core(params)
    core2.neurons = [Neuron(0, value=1.0), Neuron(1, value=1.0)]
    syn2 = core2.add_synapse(0, 1, weight=1.0)
    syn2.potential = 100.0
    nb_on = Neuronenblitz(core2, structural_plasticity_enabled=True,
                           plasticity_threshold=1.0,
                           split_probability=0.0, alternative_connection_prob=0.0,
                           backtrack_probability=0.0, backtrack_enabled=False)
    nb_on.apply_structural_plasticity([(core2.neurons[0], None), (core2.neurons[1], syn2)])
    assert len(core2.neurons) > 2


def test_loss_scale_plasticity_modulation_and_attention_scale():
    params = minimal_params()
    core = Core(params)
    core.neurons = [Neuron(0, value=1.0), Neuron(1, value=0.0)]
    syn = core.add_synapse(0, 1, weight=0.0)
    core.gradient_clip_value = 10.0
    nb = Neuronenblitz(core, loss_scale=2.0, plasticity_modulation=2.0,
                        attention_update_scale=2.0, learning_rate=1.0,
                        synapse_update_cap=10.0, attention_span_threshold=0.0,
                        split_probability=0.0, alternative_connection_prob=0.0,
                        backtrack_probability=0.0, backtrack_enabled=False)
    err = nb._compute_loss(1.0, 0.0)
    nb.apply_weight_updates_and_attention([syn], err * nb.plasticity_modulation)
    assert syn.weight == pytest.approx(1.97, abs=1e-2)
    # attention update scaling
    core_att = Core(params)
    core_att.neurons = [Neuron(0, value=1.0), Neuron(1, value=0.0)]
    syn_a = core_att.add_synapse(0, 1, weight=1.0)
    nb_att = Neuronenblitz(core_att, attention_update_scale=2.0,
                           split_probability=0.0, alternative_connection_prob=0.0,
                           backtrack_probability=0.0, backtrack_enabled=False)
    nb_att.update_attention([syn_a], error=1.0)
    nb_att2 = Neuronenblitz(core_att, attention_update_scale=1.0,
                            split_probability=0.0, alternative_connection_prob=0.0,
                            backtrack_probability=0.0, backtrack_enabled=False)
    nb_att2.update_attention([syn_a], error=1.0)
    assert nb_att.type_attention['standard'] == pytest.approx(
        2 * nb_att2.type_attention['standard']
    )
