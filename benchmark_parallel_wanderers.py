import time
import random
from marble import Neuronenblitz
from marble_core import Core


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
        'message_passing_alpha': 0.5,
        'vram_limit_mb': 0.1,
        'ram_limit_mb': 0.1,
        'disk_limit_mb': 0.1,
        'random_seed': 0,
        'attention_temperature': 1.0,
        'attention_dropout': 0.0,
        'energy_threshold': 0.0,
        'representation_noise_std': 0.0,
        'weight_init_type': 'uniform',
        'weight_init_std': 1.0,
    }


def generate_examples(n, seed=0):
    rnd = random.Random(seed)
    return [(rnd.random(), rnd.random()) for _ in range(n)]


def run_benchmark(num_examples=200):
    examples = generate_examples(num_examples)

    core1 = Core(minimal_params())
    nb1 = Neuronenblitz(core1, parallel_wanderers=1, plasticity_threshold=float('inf'))
    start = time.perf_counter()
    for ex in examples:
        nb1.train_example(*ex)
    seq_time = time.perf_counter() - start

    core2 = Core(minimal_params())
    nb2 = Neuronenblitz(core2, parallel_wanderers=2, plasticity_threshold=float('inf'))
    start = time.perf_counter()
    nb2.train_in_parallel(examples, max_workers=2)
    par_time = time.perf_counter() - start

    throughput_seq = num_examples / seq_time
    throughput_par = num_examples / par_time

    print(f"Single worker: {seq_time:.4f}s ({throughput_seq:.2f} examples/s)")
    print(f"Two workers:   {par_time:.4f}s ({throughput_par:.2f} examples/s)")


if __name__ == "__main__":
    run_benchmark()
