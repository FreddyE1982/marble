from marble_imports import *
from marble_core import Core, TIER_REGISTRY
from marble_neuronenblitz import Neuronenblitz
from neuromodulatory_system import NeuromodulatorySystem
from meta_parameter_controller import MetaParameterController
from marble_lobes import LobeManager

class Brain:
    def __init__(self, core, neuronenblitz, dataloader, save_threshold=0.05,
                 max_saved_models=5, save_dir="saved_models", firing_interval_ms=500,
                 neuromodulatory_system=None, meta_controller=None):
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
        self.best_validation_loss = float('inf')
        self.saved_model_paths = []
        self.neuromodulatory_system = neuromodulatory_system if neuromodulatory_system is not None else NeuromodulatorySystem()
        self.meta_controller = meta_controller if meta_controller is not None else MetaParameterController()
        self.lobe_manager = LobeManager(core)
        self.neurogenesis_factor = 1.0
        self.last_val_loss = None
        self.tier_decision_params = {
            'vram_usage_threshold': 0.9,
            'ram_usage_threshold': 0.9
        }
        os.makedirs(self.save_dir, exist_ok=True)

    def update_neurogenesis_factor(self, val_loss):
        """Adjust neurogenesis factor based on validation loss trends."""
        if val_loss is None:
            return
        if self.last_val_loss is None:
            self.last_val_loss = val_loss
            return
        if val_loss >= self.last_val_loss:
            self.neurogenesis_factor = min(3.0, self.neurogenesis_factor + 0.1)
        else:
            self.neurogenesis_factor = max(1.0, self.neurogenesis_factor - 0.05)
        self.last_val_loss = val_loss

    def choose_growth_tier(self):
        status = self.core.get_detailed_status()
        vram_status = status.get('vram', {})
        ram_status = status.get('ram', {})
        file_status = status.get('file', {})
        
        vram_usage = vram_status.get('memory_mb', 0)
        ram_usage = ram_status.get('memory_mb', 0)
        vram_neuron_age = vram_status.get('avg_neuron_age_sec', 0)
        ram_neuron_age = ram_status.get('avg_neuron_age_sec', 0)
        
        vram_limit = self.core.params.get('vram_limit_mb', 100)
        ram_limit = self.core.params.get('ram_limit_mb', 500)

        # Consider both memory usage and neuron age in tier selection
        if vram_usage >= vram_limit * self.tier_decision_params['vram_usage_threshold'] or vram_neuron_age > 300:  # 5 minutes
            if ram_usage >= ram_limit * self.tier_decision_params['ram_usage_threshold'] or ram_neuron_age > 600:  # 10 minutes
                if 'file' in TIER_REGISTRY:
                    chosen = "file"
                else:
                    chosen = "disk"
            else:
                chosen = "ram"
        else:
            chosen = "vram"
        
        print(f"[Brain] Growth decision: '{chosen}' tier (VRAM: {vram_usage:.2f}MB/{vram_limit}MB, age: {vram_neuron_age:.1f}s)")
        return chosen

    def perform_neurogenesis(self, base_neurons=5, base_synapses=10):
        """Grow new neurons and synapses based on neuromodulatory context."""
        ctx = self.neuromodulatory_system.get_context()
        factor = 1.0 + max(ctx.get('arousal', 0.0), ctx.get('reward', 0.0))
        factor *= self.neurogenesis_factor
        num_neurons = int(base_neurons * factor)
        num_synapses = int(base_synapses * factor)
        n_type = self.neuronenblitz.get_preferred_neuron_type()
        self.core.expand(num_new_neurons=num_neurons, num_new_synapses=num_synapses,
                         neuron_types=n_type)
        return num_neurons, num_synapses, n_type

    def train(self, train_examples, epochs=1, validation_examples=None):
        pbar = tqdm(range(epochs), desc="Epochs", ncols=100)
        for epoch in pbar:
            self.neuronenblitz.train(train_examples, epochs=1)
            self.neuronenblitz.modulate_plasticity(self.neuromodulatory_system.get_context())
            if validation_examples is not None:
                val_loss = self.validate(validation_examples)
            else:
                val_loss = None
            if val_loss is not None:
                self.meta_controller.record_loss(val_loss)
                self.meta_controller.adjust(self.neuronenblitz)
                self.update_neurogenesis_factor(val_loss)
            metrics = {
                "MeanValLoss": f"{val_loss:.4f}" if val_loss is not None else "N/A",
                "GlobalActs": self.neuronenblitz.global_activation_count,
                "VRAM(MB)": f"{self.core.get_usage_by_tier('vram'):.2f}"
            }
            pbar.set_postfix(metrics)

            if val_loss is not None and val_loss > 0.1:
                new_tier = self.choose_growth_tier()
                self.core.expand(num_new_neurons=10, num_new_synapses=15,
                               alternative_connection_prob=0.1, target_tier=new_tier)
                self.perform_neurogenesis()
            self.core.cluster_neurons(k=3)
            self.core.relocate_clusters()
            self.lobe_manager.organize()
            self.lobe_manager.self_attention(val_loss)
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

    def get_lobe_manager(self):
        """Return the lobe manager instance."""
        return self.lobe_manager

    def display_live_status(self, validation_examples):
        status = self.core.get_detailed_status()
        current_val_loss = self.validate(validation_examples)
        print("----- Live Status -----")
        for tier in status:
            print(f"{tier.upper()} -> Neurons: {status[tier]['neuron_count']}, "
                  f"Synapses: {status[tier]['synapse_count']}, "
                  f"Memory: {status[tier]['memory_mb']:.2f} MB, "
                  f"Avg Neuron Age: {status[tier]['avg_neuron_age_sec']:.1f}s")
        print(f"Current Validation Loss: {current_val_loss:.4f}")
        print(f"Global Activation Count: {self.neuronenblitz.global_activation_count}")
        print("-----------------------")

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
