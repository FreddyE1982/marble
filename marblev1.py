import cupy as cp
import numpy as np
import random
import math
import pickle
import zlib
import os
import sympy as sp
import threading
import time
from datetime import datetime
import torch
import torch.nn as nn

# ----------------------------
# 1. Basisbausteine: Neuron und Synapse
# ----------------------------
class Neuron:
    def __init__(self, nid, value=0.0, tier='vram'):
        self.id = nid
        self.value = value        # initialer Skalarwert
        self.tier = tier          # 'vram', 'ram' oder 'disk'
        self.synapses = []        # Liste ausgehender Synapsen
        self.formula = None       # Symbolische Formel (neu hinzugefügt)

class Synapse:
    def __init__(self, source, target, weight=1.0):
        self.source = source      # ID des Quellneurons
        self.target = target      # ID des Zielneurons
        self.weight = weight      # Verbindungsgewicht
        self.potential = 1.0      # Startpotential für dynamische Auswahl




# ----------------------------
# 2. Mandelbrot-Berechnung als Initialisierungsstrategie (GPU via CuPy)
# ----------------------------
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

# ----------------------------
# 3. Core – Erzeugung des neuronalen Kerns
# ----------------------------
class Core:
    def __init__(self, params, formula=None, formula_num_neurons=100):
        self.params = params
        self.vram_limit_mb = params.get('vram_limit_mb', 100)
        self.ram_limit_mb = params.get('ram_limit_mb', 500)
        self.disk_limit_mb = params.get('disk_limit_mb', 10000)
        
        self.neurons = []
        self.synapses = []
        nid = 0

        if formula is not None:
            # Symbolische Verarbeitung der Formel
            try:
                expr = sp.sympify(formula, evaluate=False)
            except Exception as e:
                raise ValueError(f"Die Formel konnte nicht geparst werden: {e}")
            
            for i_val in range(formula_num_neurons):
                neuron = Neuron(nid, value=0.0, tier='vram')
                neuron.formula = expr  # Symbolische Formel speichern
                self.neurons.append(neuron)
                nid += 1
        else:
            # Mandelbrot-Initialisierung
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

        # Verbinde benachbarte Neuronen
        num_neurons = len(self.neurons)
        for i in range(num_neurons - 1):
            weight = random.uniform(0.5, 1.5)
            syn = Synapse(self.neurons[i].id, self.neurons[i+1].id, weight)
            self.neurons[i].synapses.append(syn)
            self.synapses.append(syn)

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
        # Rückgabe eines Status-Dictionarys für die verschiedenen Tier-Ebenen
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
        print(f"Kern erweitert: {num_new_neurons} Neuronen in {new_tier} und {num_new_synapses} (+ ggf. extra) Synapsen hinzugefügt.")
        self.check_memory_usage()

# ----------------------------
# 4. DataLoader: Serialisiert, komprimiert und wandelt beliebige Daten um
# ----------------------------
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

    def load_data(self, input_data, output_data=None):
        input_tensor = self.encode(input_data)
        if output_data is not None:
            output_tensor = self.encode(output_data)
            return input_tensor, output_tensor
        return input_tensor

# ----------------------------
# 5. Neuronenblitz: Dynamisches Wandern, Backtracking und Lernmechanismen
# ----------------------------
class Neuronenblitz:
    def __init__(self, core, wander_steps=5, learning_rate=0.01,
                 backtrack_probability=0.3,
                 consolidation_probability=0.2,
                 consolidation_strength=1.1,
                 route_potential_increase=0.5,
                 route_potential_decay=0.9,
                 route_visit_decay_interval=10,
                 alternative_connection_prob=0.1):
        self.core = core
        self.wander_steps = wander_steps
        self.learning_rate = learning_rate
        self.backtrack_probability = backtrack_probability
        self.consolidation_probability = consolidation_probability
        self.consolidation_strength = consolidation_strength
        self.route_potential_increase = route_potential_increase
        self.route_potential_decay = route_potential_decay
        self.route_visit_decay_interval = route_visit_decay_interval
        self.alternative_connection_prob = alternative_connection_prob
        
        self.training_history = []
        self.global_activation_count = 0

    def relu(self, x):
        return x if x > 0 else 0

    def reset_neuron_values(self):
        for neuron in self.core.neurons:
            neuron.value = 0.0

    def weighted_choice(self, synapses):
        total = sum(syn.potential for syn in synapses)
        r = random.uniform(0, total)
        upto = 0
        for syn in synapses:
            upto += syn.potential
            if upto >= r:
                return syn
        return random.choice(synapses)

    def dynamic_wander(self, input_value):
        self.reset_neuron_values()
        vram_neurons = [n for n in self.core.neurons if n.tier == 'vram']
        entry_neuron = random.choice(vram_neurons) if vram_neurons else random.choice(self.core.neurons)
        entry_neuron.value = input_value
        current_neuron = entry_neuron
        path = [(current_neuron, None)]
        steps_taken = 0

        while steps_taken < self.wander_steps:
            if not current_neuron.synapses:
                if len(path) > 1:
                    back_steps = random.randint(1, min(2, len(path) - 1))
                    current_neuron, _ = path[-(back_steps + 1)]
                    path = path[:-back_steps]
                    continue
                else:
                    break
            if random.random() < self.backtrack_probability and len(path) > 1:
                back_steps = random.randint(1, min(2, len(path) - 1))
                current_neuron, _ = path[-(back_steps + 1)]
                path = path[:-back_steps]
                continue

            syn = self.weighted_choice(current_neuron.synapses)
            next_neuron = self.core.neurons[syn.target]
            transmitted_value = self.relu(current_neuron.value * syn.weight)
            next_neuron.value = transmitted_value
            path.append((next_neuron, syn))
            syn.potential += self.route_potential_increase
            current_neuron = next_neuron
            steps_taken += 1

        output_value = current_neuron.value
        synapse_path = [s for (_, s) in path if s is not None]
        self.global_activation_count += 1
        if self.global_activation_count % self.route_visit_decay_interval == 0:
            for syn in self.core.synapses:
                syn.potential *= self.route_potential_decay
        return output_value, synapse_path

    def train_example(self, input_value, target_value):
        output_value, path = self.dynamic_wander(input_value)
        error = target_value - output_value
        for syn in path:
            source_value = self.core.neurons[syn.source].value
            syn.weight += self.learning_rate * error * source_value
            if random.random() < self.consolidation_probability:
                syn.weight *= self.consolidation_strength
        self.training_history.append({
            'input': input_value,
            'target': target_value,
            'output': output_value,
            'error': error,
            'path_length': len(path)
        })
        return output_value, error, path

    def train(self, examples, epochs=1):
        for epoch in range(epochs):
            epoch_errors = []
            for input_val, target_val in examples:
                output, error, path = self.train_example(input_val, target_val)
                epoch_errors.append(abs(error))
            avg_error = sum(epoch_errors) / len(epoch_errors) if epoch_errors else 0
            print(f"Epoch {epoch+1}/{epochs} - Durchschnittlicher Fehler: {avg_error:.4f}")
            if avg_error > 0.1:
                self.core.expand(alternative_connection_prob=self.alternative_connection_prob)
            self.core.synapses = [s for s in self.core.synapses if abs(s.weight) >= 0.05]
    
    def get_training_history(self):
        return self.training_history

# ----------------------------
# 6. Brain – Wrapper mit DataLoader, Training, Validierung, Inference, rollierendem Model Saving und Auto-Firing
# ----------------------------
class Brain:
    def __init__(self, core, neuronenblitz, dataloader, save_threshold=0.05, max_saved_models=5, save_dir="saved_models", firing_interval_ms=None):
        self.core = core
        self.neuronenblitz = neuronenblitz
        self.dataloader = dataloader
        self.save_threshold = save_threshold
        self.max_saved_models = max_saved_models
        self.save_dir = save_dir
        self.firing_interval_ms = firing_interval_ms
        self.auto_fire_thread = None
        self.auto_fire_active = False
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_validation_loss = float('inf')
        self.saved_model_paths = []

    def train(self, train_examples, epochs=1, validation_examples=None):
        for epoch in range(epochs):
            print(f"\n--- Training Epoch {epoch+1} ---")
            self.neuronenblitz.train(train_examples, epochs=1)
            if validation_examples is not None:
                val_loss = self.validate(validation_examples)
                print(f"Validation Loss nach Epoch {epoch+1}: {val_loss:.4f}")
                self.validate_and_save(validation_examples)
                self.display_live_status(validation_examples)

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

    def start_auto_firing(self, firing_interval_ms=None, input_generator=None):
        if firing_interval_ms is not None:
            self.firing_interval_ms = firing_interval_ms
        if self.firing_interval_ms is None:
            raise ValueError("Ein Firing-Intervall in Millisekunden muss angegeben werden!")
        
        self.auto_fire_active = True
        def auto_fire_loop():
            while self.auto_fire_active:
                if input_generator is not None:
                    input_value = input_generator()
                else:
                    input_value = random.uniform(0.0, 1.0)
                output_value, path = self.neuronenblitz.dynamic_wander(input_value)
                print(f"[AutoFiring] Input: {input_value:.4f} -> Output: {output_value:.4f}, Pfadlänge: {len(path)}")
                time.sleep(self.firing_interval_ms / 1000.0)
        self.auto_fire_thread = threading.Thread(target=auto_fire_loop, daemon=True)
        self.auto_fire_thread.start()
        print(f"Auto-firing gestartet: alle {self.firing_interval_ms} ms")

    def stop_auto_firing(self):
        self.auto_fire_active = False
        if self.auto_fire_thread is not None:
            self.auto_fire_thread.join()
        print("Auto-firing gestoppt.")

    def display_live_status(self, validation_examples):
        status = self.core.get_detailed_status()
        current_val_loss = self.validate(validation_examples)
        print("----- Live Status -----")
        for tier in status:
            print(f"{tier.upper()} -> Neurons: {status[tier]['neuron_count']}, Synapses: {status[tier]['synapse_count']}, Memory: {status[tier]['memory_mb']:.2f} MB")
        print(f"Current Validation Loss: {current_val_loss:.4f}")
        print(f"Global Activation Count: {self.neuronenblitz.global_activation_count}")
        print("-----------------------")

# ----------------------------
# 7. MarbleConverter: Konvertiert ein PyTorch-Modell in ein MARBLE
# ----------------------------
class MarbleConverter:
    @staticmethod
    def convert(model, mode='sequential', core_params=None):
        if core_params is None:
            core_params = {
                'vram_limit_mb': 100,
                'ram_limit_mb': 500,
                'disk_limit_mb': 10000,
                'xmin': -2.0, 'xmax': 1.0, 'ymin': -1.5, 'ymax': 1.5,
                'width': 30, 'height': 30, 'max_iter': 50
            }
        if isinstance(model, str):
            try:
                model = torch.load(model, map_location='cpu')
            except Exception as e:
                raise ValueError(f"Fehler beim Laden des Modells: {e}")
        
        new_core = Core(core_params, formula="0", formula_num_neurons=0)
        if mode == 'sequential':
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            prev_neuron_ids = []
            for idx, layer in enumerate(layers):
                if isinstance(layer, nn.Linear):
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

# ----------------------------
# 8. Beispielhafter Ablauf
# ----------------------------
if __name__ == '__main__':
    # Parameter für Mandelbrot oder Formel
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
    
    # Core mit Formel initialisieren (Beispiel: beliebige Formel, hier: Einsteins Relativität)
    formula = "E = m * c**2"  # Beliebig austauschbar; wird symbolisch gespeichert
    core = Core(params, formula=formula, formula_num_neurons=100)
    print(f"Kern initialisiert: {len(core.neurons)} Neuronen, {len(core.synapses)} Synapsen.")
    
    # Optional: PyTorch-Modell konvertieren
    dummy_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    converted_core = MarbleConverter.convert(dummy_model, mode='sequential', core_params=params)
    
    # Brain, Neuronenblitz und DataLoader initialisieren
    nb = Neuronenblitz(core, wander_steps=7, learning_rate=0.01)
    dl = DataLoader()
    brain = Brain(core, nb, dl, save_threshold=0.05, max_saved_models=5)
    
    # Dummy-Daten für Training und Validierung
    train_examples = [(1.0, 1.5), (2.0, 2.5), (0.5, 0.8), (1.2, 1.0)]
    validation_examples = [(1.1, 1.4), (2.1, 2.6)]
    
    # Training
    brain.train(train_examples, epochs=5, validation_examples=validation_examples)
    
    # Auto-Firing starten (optional)
    brain.start_auto_firing(firing_interval_ms=2000)
    time.sleep(5)
    brain.stop_auto_firing()
