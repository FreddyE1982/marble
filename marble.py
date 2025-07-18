import os
import sys
import torch
import numpy as np
import json
import tarfile
import tempfile
import requests
import time
from pathlib import Path
from tqdm.notebook import tqdm  # For Jupyter-optimized progress bars
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
import zlib
import random
import math
import sympy as sp
import threading
from datetime import datetime
import cupy as cp
import torch.nn as nn


def clear_output(wait: bool = True) -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# -----------------------------------------------------
# 1. Logging function
# -----------------------------------------------------
def log_metrics(epoch, tar_idx, loss, vram_usage):
    with open('training_log.txt', 'a') as f:
        f.write(f"Epoch {epoch}, Tar {tar_idx}: Loss={loss:.4f}, VRAM={vram_usage:.2f}MB\n")

# -----------------------------------------------------
# 2. Download and process TAR archives (including LFS pointer detection)
# -----------------------------------------------------
def download_and_process_tar(url, temp_dir):
    print(f"Starting download from {url}")
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/91.0.4472.124 Safari/537.36')
    }
    response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    response.raise_for_status()
    content = response.content

    if len(content) < 1000 and b"oid sha256:" in content:
        print("Detected LFS pointer file. Downloading actual tar file...")
        lfs_info = content.decode("utf-8").strip().splitlines()
        oid_line = next(line for line in lfs_info if line.startswith("oid sha256:"))
        oid = oid_line.split(":")[1].strip()
        lfs_url = f"https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{url.split('data_')[-1]}"
        response = requests.get(lfs_url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading tar (LFS)", unit='iB', unit_scale=True, position=1) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        total_size = int(response.headers.get('content-length', 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading tar", unit='iB', unit_scale=True, position=1) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
    extract_dir = Path(temp_dir) / "extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError as e:
        raise ValueError(f"Failed to extract tar file: {e}")
    
    samples = []
    for root, dirs, files in os.walk(extract_dir):
        images = {}
        jsons = {}
        for file in files:
            base, ext = os.path.splitext(file)
            full_path = Path(root) / file
            if ext.lower() in ['.png', '.jpg', '.jpeg']:
                images[base] = full_path
            elif ext.lower() == '.json':
                jsons[base] = full_path
        for base in images:
            if base in jsons:
                samples.append((str(images[base]), str(jsons[base])))
    samples.sort()
    return samples

# -----------------------------------------------------
# 3. MetricsVisualizer: Live visualization of training metrics
# -----------------------------------------------------
class MetricsVisualizer:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'vram_usage': [],
            'batch_processing_time': [],
            'learning_efficiency': [],
            'memory_efficiency': []
        }
        self.setup_plot()
    
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title('MARBLE Training Metrics Live View')
        self.ax.set_xlabel('Batches')
        self.ax.set_ylabel('Loss / VRAM Usage')
        self.ax.grid(True)
    
    def update(self, new_metrics):
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        clear_output(wait=True)
        self.plot_metrics()
    
    def plot_metrics(self):
        self.ax.clear()
        self.ax.plot(self.metrics['loss'], 'b-', label='Loss')
        self.ax_twin = self.ax.twinx()
        self.ax_twin.plot(self.metrics['vram_usage'], 'r-', label='VRAM (MB)')
        self.ax.set_xlabel('Batches')
        self.ax.set_title('Training Metrics')
        self.ax.grid(True)
        self.ax.legend(loc='upper left')
        self.ax_twin.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------
# 4. MARBLE System  internal modules
# -----------------------------------------------------

# 4.1 Neuron and Synapse
class Neuron:
    def __init__(self, nid, value=0.0, tier='vram'):
        self.id = nid
        self.value = value
        self.tier = tier  # "vram", "ram", or "disk"
        self.synapses = []
        self.formula = None

class Synapse:
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        self.weight = weight
        self.potential = 1.0

# 4.2 Alternative initialization: Mandelbrot calculation (using GPU via CuPy)
def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=256):
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y
    Z = cp.zeros_like(C, dtype=cp.complex64)
    mandelbrot = cp.zeros(C.shape, dtype=cp.int32)
    for i in range(max_iter):
        mask = cp.abs(Z) <= 2
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        mandelbrot[mask] = i
    return mandelbrot

# 4.3 Core  Build the neural core (supports multiple worlds: VRAM, RAM, disk)
class Core:
    def __init__(self, params, formula=None, formula_num_neurons=100):
        print("Initializing MARBLE Core...")
        self.params = params
        self.vram_limit_mb = params.get('vram_limit_mb', 100)
        self.ram_limit_mb = params.get('ram_limit_mb', 500)
        self.disk_limit_mb = params.get('disk_limit_mb', 10000)
        self.neurons = []
        self.synapses = []
        nid = 0
        if formula is not None:
            try:
                expr = sp.sympify(formula, evaluate=False)
            except Exception as e:
                raise ValueError(f"Formula parsing failed: {e}")
            for i in range(formula_num_neurons):
                neuron = Neuron(nid, value=0.0, tier='vram')
                neuron.formula = expr
                self.neurons.append(neuron)
                nid += 1
        else:
            mandel_gpu = compute_mandelbrot(
                params['xmin'], params['xmax'],
                params['ymin'], params['ymax'],
                params['width'], params['height'],
                params.get('max_iter', 256)
            )
            mandel_cpu = cp.asnumpy(mandel_gpu)
            for val in mandel_cpu.flatten():
                self.neurons.append(Neuron(nid, value=float(val), tier='vram'))
                nid += 1

        num_neurons = len(self.neurons)
        for i in range(num_neurons - 1):
            weight = random.uniform(0.5, 1.5)
            syn = Synapse(self.neurons[i].id, self.neurons[i+1].id, weight)
            self.neurons[i].synapses.append(syn)
            self.synapses.append(syn)
        self.check_memory_usage()

    def get_usage_by_tier(self, tier):
        neurons_in_tier = [n for n in self.neurons if n.tier == tier]
        synapses_in_tier = [s for s in self.synapses if self.neurons[s.source].tier == tier]
        usage_bytes = len(neurons_in_tier) * 32 + len(synapses_in_tier) * 16
        return usage_bytes / (1024 * 1024)

    def check_memory_usage(self):
        usage_vram = self.get_usage_by_tier('vram')
        usage_ram  = self.get_usage_by_tier('ram')
        usage_disk = self.get_usage_by_tier('disk')
        print(f"Memory usage - VRAM: {usage_vram:.2f} MB, RAM: {usage_ram:.2f} MB, Disk: {usage_disk:.2f} MB")

    def get_detailed_status(self):
        status = {}
        for tier in ['vram', 'ram', 'disk']:
            neurons_in_tier = [n for n in self.neurons if n.tier == tier]
            synapses_in_tier = [s for s in self.synapses if self.neurons[s.source].tier == tier]
            status[tier] = {
                'neuron_count': len(neurons_in_tier),
                'synapse_count': len(synapses_in_tier),
                'memory_mb': self.get_usage_by_tier(tier)
            }
        return status

    def expand(self, num_new_neurons=10, num_new_synapses=15, alternative_connection_prob=0.1):
        usage_vram = self.get_usage_by_tier('vram')
        usage_ram  = self.get_usage_by_tier('ram')
        new_tier = 'vram' if usage_vram < self.vram_limit_mb else 'ram' if usage_ram < self.ram_limit_mb else 'disk'
        start_id = len(self.neurons)
        for i in range(num_new_neurons):
            self.neurons.append(Neuron(start_id + i, value=0.0, tier=new_tier))
        for _ in range(num_new_synapses):
            src = random.choice(self.neurons).id
            tgt = random.choice(self.neurons).id
            if src != tgt:
                syn = Synapse(src, tgt, weight=random.uniform(0.1, 1.0))
                self.neurons[src].synapses.append(syn)
                self.synapses.append(syn)
        print(f"Core expanded: {num_new_neurons} new neurons in {new_tier} and {num_new_synapses} new synapses added.")
        self.check_memory_usage()

# 4.4 DataLoader  Serialization and Compression
class DataLoader:
    def encode(self, data):
        serialized = pickle.dumps(data)
        compressed = zlib.compress(serialized)
        tensor = np.frombuffer(compressed, dtype=np.uint8)
        return tensor

    def decode(self, tensor):
        compressed = tensor.tobytes()
        serialized = zlib.decompress(compressed)
        data = pickle.loads(serialized)
        return data

# 4.5 Neuronenblitz  Dynamic wandering, training, structural plasticity, splitting and merging of blitz processes
class Neuronenblitz:
    def __init__(self, core,
                 backtrack_probability=0.3,
                 consolidation_probability=0.2,
                 consolidation_strength=1.1,
                 route_potential_increase=0.5,
                 route_potential_decay=0.9,
                 route_visit_decay_interval=10,
                 alternative_connection_prob=0.1,
                 split_probability=0.2,
                 merge_tolerance=0.01,
                 combine_fn=None,
                 loss_fn=None,
                 weight_update_fn=None,
                 plasticity_threshold=10.0):
        self.core = core
        self.backtrack_probability = backtrack_probability
        self.consolidation_probability = consolidation_probability
        self.consolidation_strength = consolidation_strength
        self.route_potential_increase = route_potential_increase
        self.route_potential_decay = route_potential_decay
        self.route_visit_decay_interval = route_visit_decay_interval
        self.alternative_connection_prob = alternative_connection_prob
        self.split_probability = split_probability
        self.merge_tolerance = merge_tolerance
        self.plasticity_threshold = plasticity_threshold

        self.combine_fn = combine_fn if combine_fn is not None else (lambda x, w: max(x * w, 0))
        self.loss_fn = loss_fn if loss_fn is not None else (lambda target, output: target - output)
        self.weight_update_fn = weight_update_fn if weight_update_fn is not None else (lambda source, error, path_len: (error * source) / (path_len + 1))
        
        self.training_history = []
        self.global_activation_count = 0

    def reset_neuron_values(self):
        for neuron in self.core.neurons:
            neuron.value = None

    def weighted_choice(self, synapses):
        total = sum(syn.potential for syn in synapses)
        r = random.uniform(0, total)
        upto = 0
        for syn in synapses:
            upto += syn.potential
            if upto >= r:
                return syn
        return random.choice(synapses)

    def _wander(self, current_neuron, path, current_continue_prob):
        results = []
        if not current_neuron.synapses or random.random() > current_continue_prob:
            results.append((current_neuron, path))
            return results
        if len(current_neuron.synapses) > 1 and random.random() < self.split_probability:
            for syn in current_neuron.synapses:
                next_neuron = self.core.neurons[syn.target]
                transmitted_value = self.combine_fn(current_neuron.value, syn.weight)
                next_neuron.value = transmitted_value
                new_path = path + [(next_neuron, syn)]
                new_continue_prob = current_continue_prob * 0.85
                results.extend(self._wander(next_neuron, new_path, new_continue_prob))
        else:
            syn = self.weighted_choice(current_neuron.synapses)
            next_neuron = self.core.neurons[syn.target]
            transmitted_value = self.combine_fn(current_neuron.value, syn.weight)
            next_neuron.value = transmitted_value
            new_path = path + [(next_neuron, syn)]
            new_continue_prob = current_continue_prob * 0.85
            results.extend(self._wander(next_neuron, new_path, new_continue_prob))
        return results

    def _merge_results(self, results):
        groups = {}
        for neuron, path in results:
            groups.setdefault(neuron.id, []).append((neuron, path))
        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                avg_value = sum(n.value for n, _ in group) / len(group)
                rep_path = max(group, key=lambda tup: len(tup[1]))[1]
                neuron = self.core.neurons[key]
                neuron.value = avg_value
                merged.append((neuron, rep_path))
        final_result = max(merged, key=lambda tup: tup[0].value)
        return final_result

    def dynamic_wander(self, input_value):
        for neuron in self.core.neurons:
            neuron.value = None
        entry_neuron = random.choice(self.core.neurons)
        entry_neuron.value = input_value
        initial_path = [(entry_neuron, None)]
        results = self._wander(entry_neuron, initial_path, 1.0)
        final_neuron, final_path = self._merge_results(results)
        self.global_activation_count += 1
        if self.global_activation_count % self.route_visit_decay_interval == 0:
            for syn in self.core.synapses:
                syn.potential *= self.route_potential_decay
        self.apply_structural_plasticity(final_path)
        return final_neuron.value, [s for (_, s) in final_path if s is not None]

    def apply_structural_plasticity(self, path):
        for (_, syn) in path:
            if syn is not None and syn.potential >= self.plasticity_threshold:
                source = self.core.neurons[syn.source]
                target = self.core.neurons[syn.target]
                if source.tier == 'vram':
                    new_tier = 'ram'
                elif source.tier == 'ram':
                    new_tier = 'disk'
                else:
                    new_tier = 'disk'
                new_id = len(self.core.neurons)
                new_neuron = Neuron(new_id, value=target.value, tier=new_tier)
                self.core.neurons.append(new_neuron)
                new_weight1 = syn.weight * 1.5
                new_syn1 = Synapse(source.id, new_id, weight=new_weight1)
                source.synapses.append(new_syn1)
                self.core.synapses.append(new_syn1)
                new_weight2 = syn.weight * 1.2
                new_syn2 = Synapse(new_id, target.id, weight=new_weight2)
                new_neuron.synapses.append(new_syn2)
                self.core.synapses.append(new_syn2)
                source.synapses = [s for s in source.synapses if s != syn]
                self.core.synapses = [s for s in self.core.synapses if s != syn]
                print(f"Structural plasticity: Replaced synapse from {source.id} (tier {source.tier}) to {target.id} with new neuron {new_id} in tier {new_tier}.")

    def train_example(self, input_value, target_value):
        output_value, path = self.dynamic_wander(input_value)
        error = self.loss_fn(target_value, output_value)
        path_length = len(path)
        for syn in path:
            source_value = self.core.neurons[syn.source].value
            delta = self.weight_update_fn(source_value, error, path_length)
            syn.weight += delta
            if random.random() < self.consolidation_probability:
                syn.weight *= self.consolidation_strength
        self.training_history.append({
            'input': input_value,
            'target': target_value,
            'output': output_value,
            'error': error,
            'path_length': path_length
        })
        return output_value, error, path

    def train(self, examples, epochs=1):
        for epoch in range(epochs):
            epoch_errors = []
            for input_val, target_val in examples:
                output, error, _ = self.train_example(input_val, target_val)
                epoch_errors.append(abs(error) if isinstance(error, (int, float)) else 0)
            avg_error = sum(epoch_errors) / len(epoch_errors) if epoch_errors else 0
            print(f"Epoch {epoch+1}/{epochs} - Average error: {avg_error:.4f}")
            if avg_error > 0.1:
                self.core.expand(num_new_neurons=10, num_new_synapses=15, alternative_connection_prob=self.alternative_connection_prob)
            self.core.synapses = [s for s in self.core.synapses if abs(s.weight) >= 0.05]

    def get_training_history(self):
        return self.training_history

# 4.6 Brain  Integration of training, validation, inference, model saving, auto-firing, and dreaming
class Brain:
    def __init__(self, core, neuronenblitz, dataloader, save_threshold=0.05, max_saved_models=5, save_dir="saved_models", firing_interval_ms=500):
        self.core = core
        self.neuronenblitz = neuronenblitz
        self.dataloader = dataloader
        self.save_threshold = save_threshold
        self.max_saved_models = max_saved_models
        self.save_dir = save_dir
        self.firing_interval_ms = firing_interval_ms
        self.auto_fire_thread = None
        self.auto_fire_active = False
        self.dreaming_active = False
        self.dream_thread = None

        os.makedirs(self.save_dir, exist_ok=True)
        self.best_validation_loss = float('inf')
        self.saved_model_paths = []

    def train(self, train_examples, epochs=1, validation_examples=None):
        pbar = tqdm(range(epochs), desc="Epochs", ncols=100)
        for epoch in pbar:
            self.neuronenblitz.train(train_examples, epochs=1)
            if validation_examples is not None:
                val_loss = self.validate(validation_examples)
            else:
                val_loss = None
            metrics = {
                "MeanValLoss": f"{val_loss:.4f}" if val_loss is not None else "N/A",
                "GlobalActs": self.neuronenblitz.global_activation_count,
                "VRAM(MB)": f"{self.core.get_usage_by_tier('vram'):.2f}"
            }
            pbar.set_postfix(metrics)
        pbar.close()

    def validate(self, validation_examples):
        errors = []
        for input_val, target_val in validation_examples:
            output, _ = self.neuronenblitz.dynamic_wander(input_val)
            errors.append(abs(target_val - output))
        mean_val_loss = sum(errors) / len(errors) if errors else 0
        print(f"Mean Validation Loss: {mean_val_loss:.4f}")
        return mean_val_loss

    def save_model(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brain_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump({
                'core': self.core,
                'neuronenblitz': self.neuronenblitz
            }, f)
        self.saved_model_paths.append(filepath)
        if len(self.saved_model_paths) > self.max_saved_models:
            old_file = self.saved_model_paths.pop(0)
            os.remove(old_file)
        print(f"Model saved to {filepath}")

    def validate_and_save(self, validation_examples):
        mean_val_loss = self.validate(validation_examples)
        if self.best_validation_loss - mean_val_loss >= self.save_threshold:
            self.best_validation_loss = mean_val_loss
            self.save_model()
        return mean_val_loss

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.core = data['core']
            self.neuronenblitz = data['neuronenblitz']
        print(f"Model loaded from {filepath}")

    def start_auto_firing(self, input_generator=None):
        self.auto_fire_active = True
        def auto_fire_loop():
            while self.auto_fire_active:
                if input_generator is not None:
                    input_value = input_generator()
                else:
                    input_value = random.uniform(0.0, 1.0)
                output_value, path = self.neuronenblitz.dynamic_wander(input_value)
                tqdm.write(f"[AutoFiring] Input: {input_value:.4f} -> Output: {output_value:.4f}, Path length: {len(path)}")
                time.sleep(self.firing_interval_ms / 1000.0)
        self.auto_fire_thread = threading.Thread(target=auto_fire_loop, daemon=True)
        self.auto_fire_thread.start()

    def stop_auto_firing(self):
        self.auto_fire_active = False
        if self.auto_fire_thread is not None:
            self.auto_fire_thread.join()
        print("Auto-firing stopped.")

    def dream(self, num_cycles=10):
        print("Dreaming started...")
        for cycle in range(num_cycles):
            random_input = random.uniform(0.0, 1.0)
            output, path = self.neuronenblitz.dynamic_wander(random_input)
            for syn in path:
                syn.weight *= 0.995
            print(f"Dream cycle {cycle+1}/{num_cycles}: output = {output:.4f}, path length = {len(path)}")
            time.sleep(0.1)
        print("Dreaming completed.")

    def start_dreaming(self, num_cycles=10, interval=5):
        self.dreaming_active = True
        def dream_loop():
            while self.dreaming_active:
                self.dream(num_cycles)
                time.sleep(interval)
        self.dream_thread = threading.Thread(target=dream_loop, daemon=True)
        self.dream_thread.start()

    def stop_dreaming(self):
        self.dreaming_active = False
        if self.dream_thread is not None:
            self.dream_thread.join()
        print("Dreaming stopped.")

    def display_live_status(self, validation_examples):
        status = self.core.get_detailed_status()
        current_val_loss = self.validate(validation_examples)
        print("----- Live Status -----")
        for tier in status:
            print(f"{tier.upper()} -> Neurons: {status[tier]['neuron_count']}, Synapses: {status[tier]['synapse_count']}, Memory: {status[tier]['memory_mb']:.2f} MB")
        print(f"Current Validation Loss: {current_val_loss:.4f}")
        print(f"Global Activation Count: {self.neuronenblitz.global_activation_count}")
        print("-----------------------")

# 4.7 BenchmarkManager  Compare MARBLE metrics with target metrics or a PyTorch model
class BenchmarkManager:
    def __init__(self, marble_system, target_metrics=None):
        self.marble = marble_system
        self.target_metrics = target_metrics if target_metrics is not None else {}
    
    def measure_validation_loss(self, validation_examples):
        return self.marble.get_brain().validate(validation_examples)
    
    def measure_model_size(self):
        core = self.marble.get_core()
        size = core.get_usage_by_tier('vram') + core.get_usage_by_tier('ram') + core.get_usage_by_tier('disk')
        return size
    
    def measure_inference_time(self, input_value, num_runs=10):
        times = []
        nb = self.marble.get_neuronenblitz()
        for _ in range(num_runs):
            start = time.time()
            nb.dynamic_wander(input_value)
            times.append(time.time() - start)
        return sum(times) / len(times)
    
    def compare(self, validation_examples, input_value):
        current_loss = self.measure_validation_loss(validation_examples)
        current_size = self.measure_model_size()
        current_inference_time = self.measure_inference_time(input_value)
        print("Benchmark results:")
        print(f"Validation Loss: {current_loss:.4f}")
        print(f"Model Size (MB): {current_size:.2f}")
        print(f"Inference Time (s): {current_inference_time:.4f}")
        if self.target_metrics:
            print("Target metrics:")
            for key, value in self.target_metrics.items():
                print(f"{key}: {value}")
            print("Differences:")
            if 'loss' in self.target_metrics:
                print(f"Loss diff: {current_loss - self.target_metrics['loss']:.4f}")
            if 'model_size' in self.target_metrics:
                print(f"Model size diff: {current_size - self.target_metrics['model_size']:.2f}")
            if 'inference_time' in self.target_metrics:
                print(f"Inference time diff: {current_inference_time - self.target_metrics['inference_time']:.4f}")

# 4.8 MarbleConverter  Converts a PyTorch model (e.g., text-encoder) into a MARBLE Core.
#     Extended with an option to initialize from model weights.
class MarbleConverter:
    @staticmethod
    def convert(model, mode='sequential', core_params=None, init_from_weights=False):
        if core_params is None:
            core_params = {
                'vram_limit_mb': 100,
                'ram_limit_mb': 500,
                'disk_limit_mb': 10000,
                'xmin': -2.0, 'xmax': 1.0,
                'ymin': -1.5, 'ymax': 1.5,
                'width': 30, 'height': 30,
                'max_iter': 50
            }
        if isinstance(model, str):
            try:
                model = torch.load(model, map_location='cpu')
            except Exception as e:
                raise ValueError(f"Error loading model: {e}")
        if init_from_weights:
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            if layers:
                first_layer = layers[0]
                last_layer = layers[-1]
                W_in = first_layer.weight.data.mean().item()
                W_out = last_layer.weight.data.mean().item()
                formula = f"log(1+T*{W_in:.4f})/log(1+I*{W_out:.4f})"
            else:
                formula = "0"
        else:
            formula = "0"
        new_core = Core(core_params, formula=formula, formula_num_neurons=0)
        if mode == 'sequential':
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            prev_neuron_ids = []
            for idx, layer in enumerate(layers):
                in_features = layer.in_features
                out_features = layer.out_features
                if idx == 0:
                    input_ids = list(range(len(new_core.neurons), len(new_core.neurons) + in_features))
                    for i in range(in_features):
                        new_core.neurons.append(Neuron(input_ids[i], value=0.0, tier='vram'))
                    prev_neuron_ids = input_ids
                output_ids = list(range(len(new_core.neurons), len(new_core.neurons) + out_features))
                for j in range(out_features):
                    new_core.neurons.append(Neuron(output_ids[j], value=0.0, tier='vram'))
                weight_matrix = layer.weight.detach().cpu().numpy()
                for j, out_id in enumerate(output_ids):
                    for i, in_id in enumerate(prev_neuron_ids):
                        weight_val = float(weight_matrix[j, i])
                        syn = Synapse(in_id, out_id, weight=weight_val)
                        new_core.neurons[in_id].synapses.append(syn)
                        new_core.synapses.append(syn)
                prev_neuron_ids = output_ids
        return new_core

# -----------------------------------------------------
# 5. MARBLE Class  Container for the entire MARBLE system
# -----------------------------------------------------
class MARBLE:
    """
    The MARBLE class encapsulates all internal components of the MARBLE system:
      - Core: The neural core distributed across multiple worlds (VRAM, RAM, Disk)
      - DataLoader: Serialization and compression
      - Neuronenblitz: Dynamic wandering, adaptive signal propagation, structural plasticity,
        including dynamic splitting/merging of blitz processes and inter-world migration
      - Brain: Integration of training, validation, inference, model saving, auto-firing, dreaming, and benchmarking
      - MarbleConverter: Conversion of PyTorch models (e.g., text-encoder) into a MARBLE core
      - MetricsVisualizer: Live visualization of training metrics
      - BenchmarkManager: Comparison of MARBLE metrics with target values or PyTorch models
      
    External preprocessing, postprocessing, training loops, and inference are not included in this class.
    """
    def __init__(self, params, formula=None, formula_num_neurons=100, converter_model=None, nb_params=None, brain_params=None, init_from_weights=False):
        if converter_model is not None:
            self.core = MarbleConverter.convert(converter_model, mode='sequential', core_params=params, init_from_weights=init_from_weights)
        else:
            self.core = Core(params, formula, formula_num_neurons)
        
        self.dataloader = DataLoader()
        
        nb_defaults = {
            'backtrack_probability': 0.3,
            'consolidation_probability': 0.2,
            'consolidation_strength': 1.1,
            'route_potential_increase': 0.5,
            'route_potential_decay': 0.9,
            'route_visit_decay_interval': 10,
            'alternative_connection_prob': 0.1,
            'split_probability': 0.2,
            'merge_tolerance': 0.01,
            'combine_fn': None,
            'loss_fn': None,
            'weight_update_fn': None,
            'plasticity_threshold': 10.0
        }
        if nb_params is not None:
            nb_defaults.update(nb_params)
        self.neuronenblitz = Neuronenblitz(self.core,
                                           backtrack_probability=nb_defaults['backtrack_probability'],
                                           consolidation_probability=nb_defaults['consolidation_probability'],
                                           consolidation_strength=nb_defaults['consolidation_strength'],
                                           route_potential_increase=nb_defaults['route_potential_increase'],
                                           route_potential_decay=nb_defaults['route_potential_decay'],
                                           route_visit_decay_interval=nb_defaults['route_visit_decay_interval'],
                                           alternative_connection_prob=nb_defaults['alternative_connection_prob'],
                                           split_probability=nb_defaults['split_probability'],
                                           merge_tolerance=nb_defaults['merge_tolerance'],
                                           combine_fn=nb_defaults['combine_fn'],
                                           loss_fn=nb_defaults['loss_fn'],
                                           weight_update_fn=nb_defaults['weight_update_fn'],
                                           plasticity_threshold=nb_defaults['plasticity_threshold'])
        
        brain_defaults = {
            'save_threshold': 0.05,
            'max_saved_models': 5,
            'save_dir': "saved_models",
            'firing_interval_ms': 500,
            'offload_enabled': False,
            'torrent_offload_enabled': False
        }
        if brain_params is not None:
            brain_defaults.update(brain_params)
        self.brain = Brain(self.core, self.neuronenblitz, self.dataloader,
                           save_threshold=brain_defaults['save_threshold'],
                           max_saved_models=brain_defaults['max_saved_models'],
                           save_dir=brain_defaults['save_dir'],
                           firing_interval_ms=brain_defaults['firing_interval_ms'])
        
        self.metrics_visualizer = MetricsVisualizer()
        self.benchmark_manager = BenchmarkManager(self)
    
    def get_core(self):
        return self.core

    def get_neuronenblitz(self):
        return self.neuronenblitz

    def get_brain(self):
        return self.brain

    def get_dataloader(self):
        return self.dataloader

    def get_metrics_visualizer(self):
        return self.metrics_visualizer
    
    def get_benchmark_manager(self):
        return self.benchmark_manager

# -----------------------------------------------------
# 6. BenchmarkManager  Compare MARBLE metrics with target metrics or a PyTorch model
# (Already defined above inside our consolidated code.)
# -----------------------------------------------------

# -----------------------------------------------------
# 7. MarbleConverter  Converts a PyTorch model (e.g., text-encoder) into a MARBLE Core.
# (Already defined above.)
# -----------------------------------------------------

# -----------------------------------------------------
# 8. External main part: Preprocessing, training loop, inference, benchmarking, and dreaming.
# -----------------------------------------------------
if __name__ == '__main__':
    # Core parameters
    params = {
        'xmin': -2.0,
        'xmax': 1.0,
        'ymin': -1.5,
        'ymax': 1.5,
        'width': 30,
        'height': 30,
        'max_iter': 50,
        'vram_limit_mb': 0.5,
        'ram_limit_mb': 1.0,
        'disk_limit_mb': 10
    }
    if not torch.cuda.is_available():
        params['ram_limit_mb'] += params.get('vram_limit_mb', 0)
        params['vram_limit_mb'] = 0
    formula = "log(1+T)/log(1+I)"
    from diffusers import StableDiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16
    ).to(device)
    # Instantiate the MARBLE system with the option to initialize from weights.
    marble_system = MARBLE(params, formula=formula, formula_num_neurons=100, converter_model=pipe.text_encoder, init_from_weights=True)
    core = marble_system.get_core()
    print(f"Core contains {len(core.neurons)} neurons and {len(core.synapses)} synapses.")
    
    from datasets import load_dataset
    dataset = load_dataset("laion-aesthetics-v2-5plus", split="train")
    subset_size = 10000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))
    
    # Preprocessing: For each sample, process the text prompt via the text encoder,
    # and use the colored image to compute a target scalar (sum of channel means).
    def preprocess(sample):
        caption = sample["caption"].strip()
        inputs = pipe.tokenizer(caption, return_tensors="pt")
        with torch.no_grad():
            text_embedding = pipe.text_encoder(**inputs).last_hidden_state
        input_scalar = float(text_embedding.mean().item())
        img = sample["image"]
        img_arr = np.array(img).astype(np.float32)
        if img_arr.ndim == 3 and img_arr.shape[2] == 3:
            mean_R = img_arr[:, :, 0].mean()
            mean_G = img_arr[:, :, 1].mean()
            mean_B = img_arr[:, :, 2].mean()
            target_scalar = mean_R + mean_G + mean_B
        else:
            target_scalar = img_arr.mean()
        return input_scalar, target_scalar

    train_examples = []
    val_examples = []
    for i, sample in enumerate(dataset):
        inp, tgt = preprocess(sample)
        if i % 10 == 0:
            val_examples.append((inp, tgt))
        else:
            train_examples.append((inp, tgt))
    print(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
    
    # Start auto-firing (continuous inference in the background)
    marble_system.get_brain().start_auto_firing()
    # Start dreaming (offline consolidation)
    marble_system.get_brain().start_dreaming(num_cycles=5, interval=10)
    
    # Training loop with live metrics
    num_epochs = 5
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", ncols=100)
    for epoch in epoch_pbar:
        marble_system.get_brain().train(train_examples, epochs=1, validation_examples=val_examples)
        current_val_loss = marble_system.get_brain().validate(val_examples)
        global_acts = marble_system.get_neuronenblitz().global_activation_count
        vram_usage = core.get_usage_by_tier('vram')
        epoch_pbar.set_postfix({
            "MeanValLoss": f"{current_val_loss:.4f}",
            "GlobalActs": global_acts,
            "VRAM(MB)": f"{vram_usage:.2f}"
        })
    epoch_pbar.close()
    
    marble_system.get_brain().stop_auto_firing()
    marble_system.get_brain().stop_dreaming()
    print("\nTraining completed.")
    
    benchmark_manager = marble_system.get_benchmark_manager()
    dummy_input = random.uniform(0.0, 1.0)
    benchmark_manager.compare(val_examples, dummy_input)
    
    prompt_text = "A futuristic cityscape at sunset with neon lights."
    inputs = pipe.tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = pipe.text_encoder(**inputs).last_hidden_state
    input_scalar = float(text_embedding.mean().item())
    output_scalar, path = marble_system.get_neuronenblitz().dynamic_wander(input_scalar)
    norm = (output_scalar - math.floor(output_scalar))
    color_val = int(norm * 255)
    image_array = np.full((128, 128, 3), fill_value=color_val, dtype=np.uint8)
    generated_image = Image.fromarray(image_array)
    generated_image.save("generated_image.png")
    print("Inference completed. The generated image has been saved as 'generated_image.png'.")
