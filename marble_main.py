# ruff: noqa: F401, F403, F405
import os

from marble import MarbleConverter
from marble_autograd import MarbleAutogradLayer
from marble_base import MetricsVisualizer
from marble_brain import BenchmarkManager, Brain
from marble_core import TIER_REGISTRY, Core, DataLoader
from marble_imports import *
from marble_neuronenblitz import Neuronenblitz
import torch


class MARBLE:
    def __init__(
        self,
        params,
        formula=None,
        formula_num_neurons=100,
        converter_model=None,
        nb_params=None,
        brain_params=None,
        dataloader_params=None,
        init_from_weights=False,
        remote_client=None,
        torrent_client=None,
        mv_params=None,
        dashboard_params=None,
        autograd_params=None,
        pytorch_challenge_params=None,
        hybrid_memory_params=None,
    ):
        if converter_model is not None:
            self.core = MarbleConverter.convert(
                converter_model,
                mode="sequential",
                core_params=params,
                init_from_weights=init_from_weights,
            )
        else:
            self.core = Core(params, formula, formula_num_neurons)

        mv_defaults = {
            "fig_width": 10,
            "fig_height": 6,
            "refresh_rate": 1,
            "color_scheme": "default",
            "show_neuron_ids": False,
            "dpi": 100,
            "track_memory_usage": False,
            "track_cpu_usage": False,
            "log_dir": None,
            "csv_log_path": None,
            "json_log_path": None,
            "anomaly_std_threshold": 3.0,
            "tracker": None,
        }
        if mv_params is not None:
            mv_defaults.update(mv_params)
        disable_metrics = os.environ.get("MARBLE_DISABLE_METRICS", "").lower() in (
            "1",
            "true",
        )
        self.metrics_visualizer = None
        if not disable_metrics:
            self.metrics_visualizer = MetricsVisualizer(
                fig_width=mv_defaults["fig_width"],
                fig_height=mv_defaults["fig_height"],
                refresh_rate=mv_defaults["refresh_rate"],
                color_scheme=mv_defaults["color_scheme"],
                show_neuron_ids=mv_defaults["show_neuron_ids"],
                dpi=mv_defaults["dpi"],
                track_memory_usage=mv_defaults["track_memory_usage"],
                track_cpu_usage=mv_defaults["track_cpu_usage"],
                log_dir=mv_defaults["log_dir"],
                csv_log_path=mv_defaults["csv_log_path"],
                json_log_path=mv_defaults["json_log_path"],
                anomaly_std_threshold=mv_defaults["anomaly_std_threshold"],
                tracker=mv_defaults["tracker"],
            )
        self.metrics_dashboard = None
        if (
            self.metrics_visualizer is not None
            and dashboard_params is not None
            and dashboard_params.get("enabled", False)
        ):
            from metrics_dashboard import MetricsDashboard

            self.metrics_dashboard = MetricsDashboard(
                self.metrics_visualizer,
                host=dashboard_params.get("host", "localhost"),
                port=dashboard_params.get("port", 8050),
                update_interval=dashboard_params.get("update_interval", 1000),
                window_size=dashboard_params.get("window_size", 10),
            )
            self.metrics_dashboard.start()

        dl_level = 6
        dl_enabled = True
        dl_dtype = "uint8"
        tokenizer = None
        track_meta = True
        enable_rtc = False
        rt_penalty = 0.0
        q_bits = 0
        sparse_threshold = None
        if dataloader_params is not None:
            dl_level = dataloader_params.get("compression_level", dl_level)
            dl_enabled = dataloader_params.get("compression_enabled", True)
            dl_dtype = dataloader_params.get("tensor_dtype", dl_dtype)
            track_meta = dataloader_params.get("track_metadata", True)
            enable_rtc = dataloader_params.get("enable_round_trip_check", False)
            rt_penalty = dataloader_params.get("round_trip_penalty", 0.0)
            q_bits = dataloader_params.get("quantization_bits", 0)
            sparse_threshold = dataloader_params.get("sparse_threshold")
            tok_type = dataloader_params.get("tokenizer_type")
            tok_json = dataloader_params.get("tokenizer_json")
            tok_vocab = dataloader_params.get("tokenizer_vocab_size", 30000)
            if tok_json:
                from tokenizer_utils import load_tokenizer

                tokenizer = load_tokenizer(tok_json)
            elif tok_type:
                from tokenizer_utils import built_in_tokenizer

                tokenizer = built_in_tokenizer(tok_type, vocab_size=tok_vocab)
        self.dataloader = DataLoader(
            compression_level=dl_level,
            compression_enabled=dl_enabled,
            metrics_visualizer=self.metrics_visualizer,
            tensor_dtype=dl_dtype,
            tokenizer=tokenizer,
            track_metadata=track_meta,
            enable_round_trip_check=enable_rtc,
            round_trip_penalty=rt_penalty,
            quantization_bits=q_bits,
            sparse_threshold=sparse_threshold,
        )

        nb_defaults = {
            "backtrack_probability": 0.3,
            "consolidation_probability": 0.2,
            "consolidation_strength": 1.1,
            "route_potential_increase": 0.5,
            "route_potential_decay": 0.9,
            "route_visit_decay_interval": 10,
            "alternative_connection_prob": 0.1,
            "split_probability": 0.2,
            "merge_tolerance": 0.01,
            "combine_fn": None,
            "loss_fn": None,
            "loss_module": None,
            "weight_update_fn": None,
            "plasticity_threshold": 10.0,
            "max_wander_depth": 100,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "dropout_probability": 0.0,
            "exploration_decay": 0.99,
            "reward_scale": 1.0,
            "stress_scale": 1.0,
            "remote_fallback": False,
            "noise_injection_std": 0.0,
            "dynamic_attention_enabled": True,
            "backtrack_depth_limit": 10,
            "synapse_update_cap": 1.0,
            "structural_plasticity_enabled": True,
            "backtrack_enabled": True,
            "loss_scale": 1.0,
            "exploration_bonus": 0.0,
            "synapse_potential_cap": 100.0,
            "attention_update_scale": 1.0,
            "weight_limit": 1e6,
            "wander_cache_size": 50,
            "rmsprop_beta": 0.99,
            "grad_epsilon": 1e-8,
        }
        if nb_params is not None:
            nb_defaults.update(nb_params)
        self.torrent_map = {}
        self.neuronenblitz = Neuronenblitz(
            self.core,
            remote_client=remote_client,
            torrent_client=torrent_client,
            torrent_map=self.torrent_map,
            metrics_visualizer=self.metrics_visualizer,
            **nb_defaults,
        )

        brain_defaults = {
            "save_threshold": 0.05,
            "max_saved_models": 5,
            "save_dir": "saved_models",
            "firing_interval_ms": 500,
            "offload_enabled": False,
            "torrent_offload_enabled": False,
            "mutation_rate": 0.01,
            "mutation_strength": 0.05,
            "prune_threshold": 0.01,
            "dream_num_cycles": 10,
            "dream_interval": 5,
            "neurogenesis_base_neurons": 5,
            "neurogenesis_base_synapses": 10,
            "max_training_epochs": 100,
            "memory_cleanup_enabled": True,
            "manual_seed": 0,
            "log_interval": 10,
            "evaluation_interval": 1,
            "early_stopping_patience": 5,
            "early_stopping_delta": 0.001,
            "auto_cluster_interval": 5,
            "cluster_method": "kmeans",
            "auto_save_enabled": True,
            "auto_save_interval": 5,
            "auto_firing_enabled": False,
            "dream_enabled": True,
            "vram_age_threshold": 300,
            "ram_age_threshold": 600,
            "status_display_interval": 0,
            "neurogenesis_interval": 1,
            "min_cluster_size": 1,
            "prune_frequency": 1,
            "auto_offload": False,
            "backup_enabled": False,
            "backup_interval": 3600,
            "backup_dir": "backups",
            "profile_enabled": False,
            "profile_log_path": "profile.csv",
            "profile_interval": 1,
            "benchmark_enabled": False,
            "benchmark_interval": 2,
            "model_name": "marble_default",
            "checkpoint_format": "pickle",
            "checkpoint_compress": False,
            "metrics_history_size": 100,
            "early_stop_enabled": True,
            "lobe_sync_interval": 60,
            "cleanup_batch_size": 500,
            "remote_sync_enabled": False,
            "default_activation_function": "tanh",
            "neuron_reservoir_size": 1000,
            "lobe_decay_rate": 0.98,
            "super_evolution_mode": False,
            "dream_decay_arousal_scale": 0.0,
            "dream_decay_stress_scale": 0.0,
            "dream_replay_buffer_size": 100,
            "dream_replay_batch_size": 8,
            "dream_replay_weighting": "linear",
            "dream_instant_buffer_size": 10,
            "dream_housekeeping_threshold": 0.05,
        }
        if brain_params is not None:
            brain_defaults.update(brain_params)
        ds_params = brain_defaults.pop("dimensional_search", None)
        hybrid_memory_params = brain_defaults.pop("hybrid_memory_params", None)
        self.brain = Brain(
            self.core,
            self.neuronenblitz,
            self.dataloader,
            remote_client=remote_client,
            torrent_client=torrent_client,
            torrent_map=self.torrent_map,
            metrics_visualizer=self.metrics_visualizer,
            **brain_defaults,
            dimensional_search_params=ds_params,
        )

        self.hybrid_memory = None
        if hybrid_memory_params:
            from hybrid_memory import HybridMemory

            vector_path = hybrid_memory_params.get(
                "vector_store_path", "vector_store.pkl"
            )
            symbolic_path = hybrid_memory_params.get(
                "symbolic_store_path", "symbolic_memory.pkl"
            )
            self.hybrid_memory = HybridMemory(
                self.core, self.neuronenblitz, vector_path, symbolic_path
            )

        self.benchmark_manager = BenchmarkManager(self)

        self.autograd_layer = None
        if autograd_params is not None and autograd_params.get("enabled", False):
            self.autograd_layer = MarbleAutogradLayer(
                self.brain,
                learning_rate=autograd_params.get("learning_rate", 0.01),
                accumulation_steps=autograd_params.get(
                    "gradient_accumulation_steps", 1
                ),
            )
            self.brain.set_autograd_layer(self.autograd_layer)

        self.pytorch_challenge_params = pytorch_challenge_params
        if (
            self.pytorch_challenge_params
            and self.pytorch_challenge_params.get("enabled", False)
        ):
            self.run_pytorch_challenge()

    def run_pytorch_challenge(self) -> None:
        params = self.pytorch_challenge_params or {}
        from pytorch_challenge import (
            load_dataset,
            load_pretrained_model,
            _img_to_tensor,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = load_dataset(100)
        train_data = data[:80]
        val_data = data[80:]
        model = load_pretrained_model().to(device)
        torch_inputs = [_img_to_tensor(img).to(device) for img, _ in train_data]
        marble_examples = [
            (float(img.mean()), float(lbl)) for img, lbl in train_data
        ]
        val_examples = [
            (float(img.mean()), float(lbl)) for img, lbl in val_data
        ]
        self.brain.train_pytorch_challenge(
            marble_examples,
            model,
            pytorch_inputs=torch_inputs,
            epochs=params.get("epochs", 1),
            validation_examples=val_examples,
            loss_penalty=params.get("loss_penalty", 0.1),
            speed_penalty=params.get("speed_penalty", 0.1),
            size_penalty=params.get("size_penalty", 0.1),
        )

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

    def get_autograd_layer(self):
        return self.autograd_layer

    def get_pytorch_challenge_params(self):
        return self.pytorch_challenge_params

    def __getstate__(self):
        state = self.__dict__.copy()
        state["metrics_dashboard"] = None
        state["metrics_visualizer"] = None
        state["benchmark_manager"] = None
        state["hybrid_memory"] = None
        state["autograd_layer"] = None
        state["dataloader"] = None
        state["tensor_sync_service"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.metrics_dashboard is not None:
            try:
                self.metrics_dashboard.stop()
            except Exception:
                pass
        if self.metrics_visualizer is None:
            self.metrics_visualizer = MetricsVisualizer()
        if self.dataloader is None:
            self.dataloader = DataLoader()
        if self.hybrid_memory is not None:
            try:
                self.hybrid_memory.core = self.core
                self.hybrid_memory.nb = self.neuronenblitz
            except Exception:
                pass
        if self.benchmark_manager is None:
            self.benchmark_manager = BenchmarkManager(self)
        else:
            self.benchmark_manager.marble = self
        if self.tensor_sync_service is None:
            from tensor_sync_service import TensorSyncService

            self.tensor_sync_service = TensorSyncService()


def insert_into_torch_model(
    model: torch.nn.Module,
    marble: MARBLE | None = None,
    *,
    position: int | str | None = None,
    config: dict | None = None,
    mode: str = "intermediate",
) -> tuple[torch.nn.Module, MARBLE]:
    """Insert a transparent MARBLE layer into ``model`` and return it.

    Parameters
    ----------
    model:
        PyTorch model to modify.
    marble:
        Existing MARBLE instance to attach. If ``None`` a new instance is
        created from ``config``.
    position:
        Index or name after which the layer should be inserted. ``None``
        appends the layer at the end of the network.
    config:
        Configuration dict used when ``marble`` is ``None``.
    mode:
        When ``"transparent"`` the inserted layer does not train inside the
        graph. Any other value enables in-graph training.
    """

    if marble is None:
        if config is None:
            raise ValueError("Provide either marble or config")
        marble = MARBLE(config["core"])

    train_in_graph = mode != "transparent"
    hooked = attach_marble_layer(
        model, marble, after=position, train_in_graph=train_in_graph
    )
    return hooked, marble


if __name__ == "__main__":
    # Core parameters
    params = {
        "xmin": -2.0,
        "xmax": 1.0,
        "ymin": -1.5,
        "ymax": 1.5,
        "width": 30,
        "height": 30,
        "max_iter": 50,
        "vram_limit_mb": 0.5,
        "ram_limit_mb": 1.0,
        "disk_limit_mb": 10,
    }
    if not torch.cuda.is_available():
        params["ram_limit_mb"] += params.get("vram_limit_mb", 0)
        params["vram_limit_mb"] = 0
    formula = "log(1+T)/log(1+I)"

    # Initialize MARBLE system
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to(device)

    marble_system = MARBLE(
        params,
        formula=formula,
        formula_num_neurons=100,
        converter_model=pipe.text_encoder,
        init_from_weights=True,
    )
    core = marble_system.get_core()
    print(
        f"Core contains {len(core.neurons)} neurons and {len(core.synapses)} synapses."
    )

    # Load and preprocess dataset
    from datasets import load_dataset

    dataset = load_dataset("laion-aesthetics-v2-5plus", split="train")
    subset_size = 10000
    if len(dataset) > subset_size:
        dataset = dataset.select(range(subset_size))

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
    print(
        f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}"
    )

    # Start background processes
    marble_system.get_brain().start_auto_firing()
    marble_system.get_brain().start_dreaming(num_cycles=5, interval=10)

    # Training loop with live metrics
    num_epochs = 5
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", ncols=100)
    for epoch in epoch_pbar:
        marble_system.get_brain().train(
            train_examples, epochs=1, validation_examples=val_examples
        )
        current_val_loss = marble_system.get_brain().validate(val_examples)
        global_acts = marble_system.get_neuronenblitz().global_activation_count
        vram_usage = core.get_usage_by_tier("vram")
        epoch_pbar.set_postfix(
            {
                "MeanValLoss": f"{current_val_loss:.4f}",
                "GlobalActs": global_acts,
                "VRAM(MB)": f"{vram_usage:.2f}",
            }
        )
    epoch_pbar.close()

    # Clean up background processes
    marble_system.get_brain().stop_auto_firing()
    marble_system.get_brain().stop_dreaming()
    print("\nTraining completed.")

    # Run benchmarks
    benchmark_manager = marble_system.get_benchmark_manager()
    dummy_input = random.uniform(0.0, 1.0)
    benchmark_manager.compare(val_examples, dummy_input)

    # Demonstrate inference
    prompt_text = "A futuristic cityscape at sunset with neon lights."
    inputs = pipe.tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = pipe.text_encoder(**inputs).last_hidden_state
    input_scalar = float(text_embedding.mean().item())
    output_scalar, path = marble_system.get_neuronenblitz().dynamic_wander(input_scalar)
    norm = output_scalar - math.floor(output_scalar)
    color_val = int(norm * 255)
    image_array = np.full((128, 128, 3), fill_value=color_val, dtype=np.uint8)
    generated_image = Image.fromarray(image_array)
    generated_image.save("generated_image.png")
    print(
        "Inference completed. The generated image has been saved as 'generated_image.png'."
    )
