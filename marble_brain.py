from marble_imports import *
from marble_core import TIER_REGISTRY, MemorySystem
import torch
import time
from neuromodulatory_system import NeuromodulatorySystem
from meta_parameter_controller import MetaParameterController
from marble_base import MetricsVisualizer
from marble_lobes import LobeManager


def _parse_example(sample):
    """Return ``(input_value, target_value)`` from various sample formats."""

    if isinstance(sample, dict):
        inp = (
            sample.get("input")
            or sample.get("inputs")
            or sample.get("x")
        )
        tgt = (
            sample.get("target")
            or sample.get("output")
            or sample.get("y")
            or sample.get("label")
        )
    elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
        inp, tgt = sample[0], sample[1]
    else:
        raise TypeError("Unsupported sample type")

    if torch.is_tensor(inp):
        inp = inp.detach().cpu().float().mean().item()
    elif isinstance(inp, np.ndarray):
        inp = float(np.mean(inp))

    if torch.is_tensor(tgt):
        tgt = tgt.detach().cpu().float().mean().item()
    elif isinstance(tgt, np.ndarray):
        tgt = float(np.mean(tgt))

    return float(inp), float(tgt)


def _normalize_examples(examples):
    """Convert supported datasets or iterables to a list of ``(input, target)``."""

    if isinstance(examples, list):
        return [_parse_example(e) for e in examples]

    try:
        return [_parse_example(e) for e in examples]
    except TypeError:
        raise


class Brain:
    def __init__(
        self,
        core,
        neuronenblitz,
        dataloader,
        save_threshold=0.05,
        max_saved_models=5,
        save_dir="saved_models",
        firing_interval_ms=500,
        neuromodulatory_system=None,
        meta_controller=None,
        memory_system=None,
        remote_client=None,
        torrent_client=None,
        torrent_map=None,
        autograd_layer=None,
        tier_decision_params=None,
        initial_neurogenesis_factor: float = 1.0,
        offload_enabled: bool = False,
        torrent_offload_enabled: bool = False,
        mutation_rate: float = 0.01,
        mutation_strength: float = 0.05,
        prune_threshold: float = 0.01,
        dream_num_cycles: int = 10,
        dream_interval: int = 5,
        neurogenesis_base_neurons: int = 5,
        neurogenesis_base_synapses: int = 10,
        max_training_epochs: int = 100,
        memory_cleanup_enabled: bool = True,
        manual_seed: int = 0,
        log_interval: int = 10,
        evaluation_interval: int = 1,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 0.001,
        auto_cluster_interval: int = 5,
        cluster_method: str = "kmeans",
        auto_save_enabled: bool = True,
        offload_threshold: float = 1.0,
        torrent_offload_threshold: float = 1.0,
        cluster_high_threshold: float = 1.0,
        cluster_medium_threshold: float = 0.1,
        dream_synapse_decay: float = 0.995,
        dream_decay_arousal_scale: float = 0.0,
        dream_decay_stress_scale: float = 0.0,
        neurogenesis_increase_step: float = 0.1,
        neurogenesis_decrease_step: float = 0.05,
        max_neurogenesis_factor: float = 3.0,
        cluster_k: int = 3,
        auto_save_interval: int = 5,
        auto_firing_enabled: bool = False,
        dream_enabled: bool = True,
        vram_age_threshold: int = 300,
        ram_age_threshold: int = 600,
        status_display_interval: int = 0,
        neurogenesis_interval: int = 1,
        min_cluster_size: int = 1,
        prune_frequency: int = 1,
        auto_offload: bool = False,
        benchmark_enabled: bool = False,
        benchmark_interval: int = 2,
        loss_growth_threshold: float = 0.1,
        auto_neurogenesis_prob: float = 0.0,
        dream_cycle_sleep: float = 0.1,
        lobe_attention_increase: float = 1.05,
        lobe_attention_decrease: float = 0.95,
        model_name: str = "marble_default",
        checkpoint_format: str = "pickle",
        metrics_history_size: int = 100,
        early_stop_enabled: bool = True,
        lobe_sync_interval: int = 60,
        cleanup_batch_size: int = 500,
        remote_sync_enabled: bool = False,
        default_activation_function: str = "tanh",
        neuron_reservoir_size: int = 1000,
        lobe_decay_rate: float = 0.98,
        super_evolution_mode: bool = False,
        metrics_visualizer=None,
        dimensional_search_params=None,
    ):
        self.core = core
        self.neuronenblitz = neuronenblitz
        self.dataloader = dataloader
        self.save_threshold = save_threshold
        self.max_saved_models = max_saved_models
        self.save_dir = save_dir
        self.firing_interval_ms = firing_interval_ms
        self.auto_fire_thread = None
        self.auto_fire_active = False
        self.training_thread = None
        self.training_active = False
        self.dreaming_active = False
        self.dream_thread = None
        self.best_validation_loss = float("inf")
        self.saved_model_paths = []
        self.last_chain_of_thought = []
        self.neuromodulatory_system = (
            neuromodulatory_system
            if neuromodulatory_system is not None
            else NeuromodulatorySystem()
        )
        self.meta_controller = (
            meta_controller
            if meta_controller is not None
            else MetaParameterController()
        )
        self.memory_system = (
            memory_system if memory_system is not None else MemorySystem()
        )
        self.lobe_manager = LobeManager(
            core,
            attention_increase_factor=lobe_attention_increase,
            attention_decrease_factor=lobe_attention_decrease,
        )
        self.neurogenesis_factor = initial_neurogenesis_factor
        self.remote_client = remote_client
        self.torrent_client = torrent_client
        self.torrent_map = torrent_map if torrent_map is not None else {}
        self.autograd_layer = autograd_layer
        self.offload_enabled = offload_enabled
        self.torrent_offload_enabled = torrent_offload_enabled
        self.dream_num_cycles = dream_num_cycles
        self.dream_interval = dream_interval
        self.neurogenesis_base_neurons = neurogenesis_base_neurons
        self.neurogenesis_base_synapses = neurogenesis_base_synapses
        self.max_training_epochs = max_training_epochs
        self.memory_cleanup_enabled = memory_cleanup_enabled
        self.manual_seed = manual_seed
        self.log_interval = log_interval
        self.evaluation_interval = evaluation_interval
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.auto_cluster_interval = auto_cluster_interval
        self.cluster_method = cluster_method
        self.auto_save_enabled = auto_save_enabled
        self.offload_threshold = offload_threshold
        self.torrent_offload_threshold = torrent_offload_threshold
        self.cluster_high_threshold = cluster_high_threshold
        self.cluster_medium_threshold = cluster_medium_threshold
        self.dream_synapse_decay = dream_synapse_decay
        self.dream_decay_arousal_scale = dream_decay_arousal_scale
        self.dream_decay_stress_scale = dream_decay_stress_scale
        self.neurogenesis_increase_step = neurogenesis_increase_step
        self.neurogenesis_decrease_step = neurogenesis_decrease_step
        self.max_neurogenesis_factor = max_neurogenesis_factor
        self.cluster_k = cluster_k
        self.auto_save_interval = auto_save_interval
        self.auto_firing_enabled = auto_firing_enabled
        self.dream_enabled = dream_enabled
        self.vram_age_threshold = vram_age_threshold
        self.ram_age_threshold = ram_age_threshold
        self.status_display_interval = status_display_interval
        self.neurogenesis_interval = neurogenesis_interval
        self.min_cluster_size = min_cluster_size
        self.prune_frequency = prune_frequency
        self.auto_offload = auto_offload
        self.benchmark_enabled = benchmark_enabled
        self.benchmark_interval = benchmark_interval
        self._benchmark_counter = 0
        self.loss_growth_threshold = loss_growth_threshold
        self.auto_neurogenesis_prob = auto_neurogenesis_prob
        self.dream_cycle_sleep = dream_cycle_sleep
        self.model_name = model_name
        self.checkpoint_format = checkpoint_format
        self.metrics_history_size = metrics_history_size
        self.early_stop_enabled = early_stop_enabled
        self.lobe_sync_interval = lobe_sync_interval
        self.cleanup_batch_size = cleanup_batch_size
        self.remote_sync_enabled = remote_sync_enabled
        self.default_activation_function = default_activation_function
        self.neuron_reservoir_size = neuron_reservoir_size
        self.lobe_decay_rate = lobe_decay_rate
        self.super_evolution_mode = super_evolution_mode
        self.super_evo_controller = None
        if self.super_evolution_mode:
            from super_evolution_controller import SuperEvolutionController

            self.super_evo_controller = SuperEvolutionController(self)
        self.last_val_loss = None
        self.tier_decision_params = (
            tier_decision_params
            if tier_decision_params is not None
            else {"vram_usage_threshold": 0.9, "ram_usage_threshold": 0.9}
        )
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.prune_threshold = prune_threshold
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_visualizer = metrics_visualizer
        self.dim_search = None
        if dimensional_search_params is not None and dimensional_search_params.get("enabled", False):
            from dimensional_search import DimensionalitySearch
            self.dim_search = DimensionalitySearch(
                self.core,
                max_size=dimensional_search_params.get("max_size", self.core.rep_size),
                improvement_threshold=dimensional_search_params.get("improvement_threshold", 0.02),
                plateau_epochs=dimensional_search_params.get("plateau_epochs", 2),
                metrics_visualizer=self.metrics_visualizer,
            )

    def set_autograd_layer(self, layer):
        """Attach an autograd layer for benchmarking."""
        self.autograd_layer = layer

    def update_neurogenesis_factor(self, val_loss):
        """Adjust neurogenesis factor based on validation loss trends."""
        if val_loss is None:
            return
        if self.last_val_loss is None:
            self.last_val_loss = val_loss
            return
        if val_loss >= self.last_val_loss:
            self.neurogenesis_factor = min(
                self.max_neurogenesis_factor,
                self.neurogenesis_factor + self.neurogenesis_increase_step,
            )
        else:
            self.neurogenesis_factor = max(
                1.0,
                self.neurogenesis_factor - self.neurogenesis_decrease_step,
            )
        self.last_val_loss = val_loss

    def compute_dream_decay(self) -> float:
        """Return synapse decay factor adjusted by neuromodulatory signals."""
        ctx = self.neuromodulatory_system.get_context()
        arousal = ctx.get("arousal", 0.0)
        stress = ctx.get("stress", 0.0)
        decay = self.dream_synapse_decay
        decay *= 1.0 + self.dream_decay_arousal_scale * arousal
        decay *= 1.0 - self.dream_decay_stress_scale * stress
        return max(0.0, min(decay, 1.0))

    def choose_growth_tier(self):
        status = self.core.get_detailed_status()
        vram_status = status.get("vram", {})
        ram_status = status.get("ram", {})
        file_status = status.get("file", {})

        vram_usage = vram_status.get("memory_mb", 0)
        ram_usage = ram_status.get("memory_mb", 0)
        vram_neuron_age = vram_status.get("avg_neuron_age_sec", 0)
        ram_neuron_age = ram_status.get("avg_neuron_age_sec", 0)

        vram_limit = self.core.params.get("vram_limit_mb", 100)
        ram_limit = self.core.params.get("ram_limit_mb", 500)

        # Consider both memory usage and neuron age in tier selection
        if (
            vram_usage >= vram_limit * self.tier_decision_params["vram_usage_threshold"]
            or vram_neuron_age > 300
        ):  # 5 minutes
            if (
                ram_usage
                >= ram_limit * self.tier_decision_params["ram_usage_threshold"]
                or ram_neuron_age > 600
            ):  # 10 minutes
                if "file" in TIER_REGISTRY:
                    chosen = "file"
                else:
                    chosen = "disk"
            else:
                chosen = "ram"
        else:
            chosen = "vram"

        print(
            f"[Brain] Growth decision: '{chosen}' tier (VRAM: {vram_usage:.2f}MB/{vram_limit}MB, age: {vram_neuron_age:.1f}s)"
        )
        return chosen

    def perform_neurogenesis(
        self, base_neurons=None, base_synapses=None, use_combined_attention=False
    ):
        """Grow new neurons and synapses based on neuromodulatory context."""
        if base_neurons is None:
            base_neurons = self.neurogenesis_base_neurons
        if base_synapses is None:
            base_synapses = self.neurogenesis_base_synapses
        ctx = self.neuromodulatory_system.get_context()
        factor = 1.0 + max(ctx.get("arousal", 0.0), ctx.get("reward", 0.0))
        factor *= self.neurogenesis_factor
        num_neurons = int(base_neurons * factor)
        num_synapses = int(base_synapses * factor)
        if use_combined_attention:
            n_type = self.neuronenblitz.get_combined_preferred_neuron_type()
        else:
            n_type = self.neuronenblitz.get_preferred_neuron_type()
        self.core.expand(
            num_new_neurons=num_neurons,
            num_new_synapses=num_synapses,
            neuron_types=n_type,
        )
        return num_neurons, num_synapses, n_type

    def maybe_autonomous_neurogenesis(self, val_loss=None):
        prob = self.auto_neurogenesis_prob
        if val_loss is not None:
            prob *= min(1.0, float(val_loss))
        if random.random() < prob:
            self.perform_neurogenesis(use_combined_attention=True)
            return True
        return False

    def train(self, train_examples, epochs=1, validation_examples=None):
        train_examples = _normalize_examples(train_examples)
        validation_examples = (
            _normalize_examples(validation_examples)
            if validation_examples is not None
            else None
        )
        pbar = tqdm(range(epochs), desc="Epochs", ncols=100)
        best_loss = float("inf")
        patience_counter = 0
        for epoch in pbar:
            start_time = time.time()
            self.neuronenblitz.train(train_examples, epochs=1)
            self.neuronenblitz.modulate_plasticity(
                self.neuromodulatory_system.get_context()
            )
            if validation_examples is not None:
                val_loss = self.validate(validation_examples)
            else:
                val_loss = None
            if val_loss is not None:
                self.meta_controller.record_loss(val_loss)
                self.meta_controller.adjust(self.neuronenblitz)
                self.update_neurogenesis_factor(val_loss)
                self.maybe_autonomous_neurogenesis(val_loss)
                if self.early_stop_enabled:
                    if val_loss < best_loss - self.early_stopping_delta:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            pbar.close()
                            print("Early stopping triggered")
                            break
            metrics = {
                "MeanValLoss": f"{val_loss:.4f}" if val_loss is not None else "N/A",
                "GlobalActs": self.neuronenblitz.global_activation_count,
                "VRAM(MB)": f"{self.core.get_usage_by_tier('vram'):.2f}",
            }
            pbar.set_postfix(metrics)
            if self.metrics_visualizer is not None:
                ctx = self.neuromodulatory_system.get_context()
                self.metrics_visualizer.update(
                    {
                        "loss": val_loss if val_loss is not None else 0.0,
                        "vram_usage": self.core.get_usage_by_tier("vram"),
                        "arousal": ctx.get("arousal", 0.0),
                        "stress": ctx.get("stress", 0.0),
                        "reward": ctx.get("reward", 0.0),
                        "plasticity_threshold": self.neuronenblitz.plasticity_threshold,
                        "message_passing_change": self.neuronenblitz.last_message_passing_change,
                        "meta_loss_avg": (
                            sum(self.meta_controller.loss_history)
                            / len(self.meta_controller.loss_history)
                            if self.meta_controller.loss_history
                            else 0.0
                        ),
                    }
                )
            if self.dim_search is not None and val_loss is not None:
                self.dim_search.evaluate(val_loss)

            epoch_time = time.time() - start_time
            if self.super_evo_controller is not None:
                self.super_evo_controller.record_metrics(
                    0.0 if val_loss is None else val_loss,
                    epoch_time,
                )

            if val_loss is not None and val_loss > self.loss_growth_threshold:
                new_tier = self.choose_growth_tier()
                self.core.expand(
                    num_new_neurons=10,
                    num_new_synapses=15,
                    alternative_connection_prob=0.1,
                    target_tier=new_tier,
                )
                self.perform_neurogenesis(use_combined_attention=True)
            self.core.cluster_neurons(k=self.cluster_k)
            self.core.relocate_clusters(
                high=self.cluster_high_threshold,
                medium=self.cluster_medium_threshold,
            )
            self.lobe_manager.organize()
            self.lobe_manager.self_attention(val_loss)
            if self.offload_enabled:
                self.offload_high_attention(self.offload_threshold)
            if self.torrent_offload_enabled:
                self.offload_high_attention_torrent(self.torrent_offload_threshold)
            self.consolidate_memory()
            self.evolve()
            self._benchmark_counter += 1
            if (
                self.benchmark_enabled
                and self.autograd_layer is not None
                and self._benchmark_counter % self.benchmark_interval == 0
                and train_examples
            ):
                example = random.choice(train_examples)
                self.benchmark_step(example)
        pbar.close()

    def validate(self, validation_examples):
        validation_examples = _normalize_examples(validation_examples)
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
            pickle.dump({"core": self.core, "neuronenblitz": self.neuronenblitz}, f)
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
            self.core = data["core"]
            self.neuronenblitz = data["neuronenblitz"]
        print(f"Model loaded from {filepath}")

    def infer(self, input_value):
        """Return the output of the trained model for ``input_value``."""
        output, _ = self.neuronenblitz.dynamic_wander(float(input_value))
        return float(output)

    def generate_chain_of_thought(self, input_value):
        """Return output and a chain of reasoning steps for the given input."""
        output, path = self.neuronenblitz.dynamic_wander(input_value)
        chain = []
        prev_val = input_value
        for syn in path:
            to_val = self.core.neurons[syn.target].value
            chain.append(
                {
                    "from": syn.source,
                    "to": syn.target,
                    "weight": syn.weight,
                    "input": prev_val,
                    "output": to_val,
                }
            )
            prev_val = to_val
        self.last_chain_of_thought = chain
        return output, chain

    def store_memory(self, key, value):
        layer = self.memory_system.choose_layer(
            self.neuromodulatory_system.get_context()
        )
        layer.store(key, value)

    def retrieve_memory(self, key):
        val = self.memory_system.short_term.retrieve(key)
        if val is None:
            val = self.memory_system.long_term.retrieve(key)
        return val

    def consolidate_memory(self):
        self.memory_system.consolidate()

    def start_training(self, train_examples, epochs=1, validation_examples=None):
        """Begin training in a background thread."""
        if self.training_active:
            return
        self.training_active = True

        def train_loop():
            try:
                self.train(
                    train_examples,
                    epochs=epochs,
                    validation_examples=validation_examples,
                )
            except Exception as e:
                print(f"Training thread error: {e}")
            finally:
                self.training_active = False

        self.training_thread = threading.Thread(target=train_loop, daemon=True)
        self.training_thread.start()

    def wait_for_training(self):
        """Block until background training finishes."""
        if self.training_thread is not None:
            self.training_thread.join()

    def start_auto_firing(self, input_generator=None):
        self.auto_fire_active = True

        def auto_fire_loop():
            while self.auto_fire_active:
                if input_generator is not None:
                    input_value = input_generator()
                else:
                    input_value = random.uniform(0.0, 1.0)
                output_value, path = self.neuronenblitz.dynamic_wander(input_value)
                tqdm.write(
                    f"[AutoFiring] Input: {input_value:.4f} -> Output: {output_value:.4f}, Path length: {len(path)}"
                )
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
                syn.weight *= self.compute_dream_decay()
            print(
                f"Dream cycle {cycle+1}/{num_cycles}: output = {output:.4f}, path length = {len(path)}"
            )
            time.sleep(self.dream_cycle_sleep)
        print("Dreaming completed.")

    def start_dreaming(self, num_cycles=None, interval=None):
        if num_cycles is None:
            num_cycles = self.dream_num_cycles
        if interval is None:
            interval = self.dream_interval
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

    def mutate_synapses(self, mutation_rate=0.01, mutation_strength=0.05):
        """Randomly adjust synapse weights to introduce variation."""
        mutated = 0
        for syn in self.core.synapses:
            if random.random() < mutation_rate:
                syn.weight += random.uniform(-mutation_strength, mutation_strength)
                mutated += 1
        return mutated

    def prune_weak_synapses(self, threshold=0.01):
        """Remove synapses with very small absolute weights."""
        pruned = 0
        self.core.synapses = [
            s for s in self.core.synapses if abs(s.weight) >= threshold
        ]
        for neuron in self.core.neurons:
            before = len(neuron.synapses)
            neuron.synapses = [s for s in neuron.synapses if abs(s.weight) >= threshold]
            pruned += before - len(neuron.synapses)
        return pruned

    def evolve(self, mutation_rate=None, mutation_strength=None, prune_threshold=None):
        """Apply evolutionary operators like mutation and pruning."""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        if mutation_strength is None:
            mutation_strength = self.mutation_strength
        if prune_threshold is None:
            prune_threshold = self.prune_threshold
        mutated = self.mutate_synapses(mutation_rate, mutation_strength)
        pruned = self.prune_weak_synapses(prune_threshold)
        return mutated, pruned

    def get_lobe_manager(self):
        """Return the lobe manager instance."""
        return self.lobe_manager

    def offload_high_attention(self, threshold=1.0):
        if not self.offload_enabled or self.remote_client is None:
            return
        ids = self.lobe_manager.select_high_attention(threshold)
        if not ids:
            return
        subcore = self.core.extract_subcore(ids)
        for nid in ids:
            self.core.neurons[nid].tier = "remote"
        self.remote_client.offload(subcore)

    def offload_high_attention_torrent(self, threshold=1.0):
        if not self.torrent_offload_enabled or self.torrent_client is None:
            return
        ids = self.lobe_manager.select_high_attention(threshold)
        if not ids:
            return
        subcore = self.core.extract_subcore(ids)
        part = self.torrent_client.offload(subcore)
        for nid in ids:
            self.core.neurons[nid].tier = "torrent"
            self.torrent_map[nid] = part

    def display_live_status(self, validation_examples):
        status = self.core.get_detailed_status()
        current_val_loss = self.validate(validation_examples)
        print("----- Live Status -----")
        for tier in status:
            print(
                f"{tier.upper()} -> Neurons: {status[tier]['neuron_count']}, "
                f"Synapses: {status[tier]['synapse_count']}, "
                f"Memory: {status[tier]['memory_mb']:.2f} MB, "
                f"Avg Neuron Age: {status[tier]['avg_neuron_age_sec']:.1f}s"
            )
        print(f"Current Validation Loss: {current_val_loss:.4f}")
        print(f"Global Activation Count: {self.neuronenblitz.global_activation_count}")
        print("-----------------------")

    def benchmark_step(self, example):
        """Run a benchmark comparison on a single (input, target) pair."""
        if self.autograd_layer is None:
            return None

        input_val, target_val = example

        start = time.time()
        _, error, _ = self.neuronenblitz.train_example(input_val, target_val)
        marble_time = time.time() - start
        marble_loss = abs(error) if isinstance(error, (int, float)) else 0.0

        start = time.time()
        inp = torch.tensor(float(input_val), dtype=torch.float32, requires_grad=True)
        out = self.autograd_layer(inp)
        loss = (out - torch.tensor(float(target_val), dtype=torch.float32)) ** 2
        loss_val = float(loss.item())
        loss.backward()
        auto_time = time.time() - start

        print(
            f"[Benchmark] Marble loss {marble_loss:.4f} time {marble_time:.4f}s | "
            f"Autograd loss {loss_val:.4f} time {auto_time:.4f}s"
        )

        if marble_loss > loss_val:
            self.neuronenblitz.learning_rate *= 1.1
        if marble_time > auto_time:
            self.neuronenblitz.continue_decay_rate *= 1.05

        return {
            "marble": {"loss": marble_loss, "time": marble_time},
            "autograd": {"loss": loss_val, "time": auto_time},
        }

    def train_pytorch_challenge(
        self,
        train_examples,
        pytorch_model,
        pytorch_inputs=None,
        epochs=1,
        validation_examples=None,
        loss_penalty=0.1,
        speed_penalty=0.1,
        size_penalty=0.1,
    ):
        """Train with penalties relative to a PyTorch model."""
        pyro_size = sum(p.numel() for p in pytorch_model.parameters()) * 4 / 1e6
        pytorch_model.eval()
        pbar = tqdm(range(epochs), desc="ChallengeEpochs", ncols=100)
        if pytorch_inputs is None:
            pytorch_inputs = [
                torch.tensor(inp, dtype=torch.float32) for inp, _ in train_examples
            ]
        for _ in pbar:
            for (inp, tgt), p_inp in zip(train_examples, pytorch_inputs):
                start = time.time()
                _, err, _ = self.neuronenblitz.train_example(float(inp), float(tgt))
                marble_time = time.time() - start
                marble_loss = abs(err) if isinstance(err, (int, float)) else 0.0
                tensor = p_inp
                start = time.time()
                with torch.no_grad():
                    py_out = pytorch_model(tensor)
                pyro_time = time.time() - start
                pyro_loss = float(abs(float(tgt) - float(py_out.flatten()[0])))

                penalty = 0.0
                if marble_loss > pyro_loss:
                    penalty += loss_penalty
                if marble_time > pyro_time:
                    penalty += speed_penalty
                marble_size = (
                    self.core.get_usage_by_tier("vram")
                    + self.core.get_usage_by_tier("ram")
                    + self.core.get_usage_by_tier("disk")
                )
                if marble_size > pyro_size:
                    penalty += size_penalty
                ctx = self.neuromodulatory_system.get_context()
                new_stress = min(1.0, ctx.get("stress", 0.0) + penalty)
                self.neuromodulatory_system.update_signals(stress=new_stress)
                self.neuronenblitz.modulate_plasticity(
                    self.neuromodulatory_system.get_context()
                )
            if validation_examples is not None:
                val_loss = self.validate(validation_examples)
                pbar.set_postfix({"ValLoss": f"{val_loss:.4f}"})
        pbar.close()


    def __getstate__(self):
        state = self.__dict__.copy()
        state["training_thread"] = None
        state["auto_fire_thread"] = None
        state["dream_thread"] = None
        state["metrics_visualizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.training_thread = None
        self.auto_fire_thread = None
        self.dream_thread = None
        if self.metrics_visualizer is None:
            from marble_base import MetricsVisualizer

            self.metrics_visualizer = MetricsVisualizer()

class BenchmarkManager:
    def __init__(self, marble_system, target_metrics=None):
        self.marble = marble_system
        self.target_metrics = target_metrics if target_metrics is not None else {}

    def measure_validation_loss(self, validation_examples):
        return self.marble.get_brain().validate(validation_examples)

    def measure_model_size(self):
        core = self.marble.get_core()
        size = (
            core.get_usage_by_tier("vram")
            + core.get_usage_by_tier("ram")
            + core.get_usage_by_tier("disk")
        )
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
            if "loss" in self.target_metrics:
                print(f"Loss diff: {current_loss - self.target_metrics['loss']:.4f}")
            if "model_size" in self.target_metrics:
                print(
                    f"Model size diff: {current_size - self.target_metrics['model_size']:.2f}"
                )
            if "inference_time" in self.target_metrics:
                print(
                    f"Inference time diff: {current_inference_time - self.target_metrics['inference_time']:.4f}"
                )
