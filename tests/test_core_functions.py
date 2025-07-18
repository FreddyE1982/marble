import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import random
from marble_imports import cp
from marble_core import compute_mandelbrot, DataLoader, Core
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain


def minimal_params():
    return {
        'xmin': -2.0,
        'xmax': 1.0,
        'ymin': -1.5,
        'ymax': 1.5,
        'width': 3,
        'height': 3,
        'max_iter': 5,
        'representation_size': 4,
        'vram_limit_mb': 0.1,
        'ram_limit_mb': 0.1,
        'disk_limit_mb': 0.1,
    }


def test_compute_mandelbrot_shape():
    arr = compute_mandelbrot(-2, 1, -1.5, 1.5, 4, 4, max_iter=5)
    np_arr = cp.asnumpy(arr)
    assert np_arr.shape == (4, 4)
    assert np_arr.dtype == np.int32


def test_dataloader_roundtrip():
    dl = DataLoader()
    data = {'a': 1, 'b': [1, 2, 3]}
    tensor = dl.encode(data)
    out = dl.decode(tensor)
    assert out == data


def test_dataloader_array_roundtrip():
    dl = DataLoader()
    arr = np.arange(9, dtype=np.int32).reshape(3, 3)
    tensor = dl.encode_array(arr)
    restored = dl.decode_array(tensor)
    assert np.array_equal(restored, arr)


def test_core_expand_adds_neurons():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    initial_neurons = len(core.neurons)
    core.expand(num_new_neurons=2, num_new_synapses=2)
    assert len(core.neurons) >= initial_neurons + 2

def test_core_expand_assigns_types():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    core.expand(num_new_neurons=3, num_new_synapses=0, neuron_types=['excitatory'])
    types = {n.neuron_type for n in core.neurons[-3:]}
    assert types == {'excitatory'}


def test_neuronenblitz_train_example_updates_history():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    output, error, path = nb.train_example(0.5, 0.2)
    assert isinstance(output, float)
    assert isinstance(error, float)
    assert nb.training_history


def test_brain_validate_runs():
    random.seed(0)
    params = minimal_params()
    core = Core(params)
    nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())
    examples = [(0.1, 0.2), (0.2, 0.3)]
    nb.train(examples, epochs=1)
    val_loss = brain.validate(examples)
    assert isinstance(val_loss, float)
