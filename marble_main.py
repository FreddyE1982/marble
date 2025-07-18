from marble_imports import *
from marble_core import Core, DataLoader, TIER_REGISTRY
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain, BenchmarkManager
from marble_base import MetricsVisualizer

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
    ):
        if converter_model is not None:
            self.core = MarbleConverter.convert(converter_model, mode='sequential', core_params=params, init_from_weights=init_from_weights)
        else:
            self.core = Core(params, formula, formula_num_neurons)

        mv_defaults = {"fig_width": 10, "fig_height": 6, "refresh_rate": 1, "color_scheme": "default", "show_neuron_ids": False, "dpi": 100}
        if mv_params is not None:
            mv_defaults.update(mv_params)
        self.metrics_visualizer = MetricsVisualizer(
            fig_width=mv_defaults["fig_width"],
            fig_height=mv_defaults["fig_height"],
        )
        
        dl_level = 6
        if dataloader_params is not None:
            dl_level = dataloader_params.get("compression_level", dl_level)
        self.dataloader = DataLoader(
            compression_level=dl_level,
            metrics_visualizer=self.metrics_visualizer,
        )
        
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
            'plasticity_threshold': 10.0,
            'max_wander_depth': 100,
            'learning_rate': 0.01,
            'weight_decay': 0.0,
            'dropout_probability': 0.0,
            'exploration_decay': 0.99,
            'reward_scale': 1.0,
            'stress_scale': 1.0,
            'remote_fallback': False,
            'noise_injection_std': 0.0,
            'dynamic_attention_enabled': True,
            'backtrack_depth_limit': 10,
            'synapse_update_cap': 1.0,
            'structural_plasticity_enabled': True,
            'backtrack_enabled': True,
            'loss_scale': 1.0,
            'exploration_bonus': 0.0,
            'synapse_potential_cap': 100.0,
            'attention_update_scale': 1.0,
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
            'save_threshold': 0.05,
            'max_saved_models': 5,
            'save_dir': "saved_models",
            'firing_interval_ms': 500,
            'offload_enabled': False,
            'torrent_offload_enabled': False,
            'mutation_rate': 0.01,
            'mutation_strength': 0.05,
            'prune_threshold': 0.01,
            'dream_num_cycles': 10,
            'dream_interval': 5,
            'neurogenesis_base_neurons': 5,
            'neurogenesis_base_synapses': 10,
            'max_training_epochs': 100,
            'memory_cleanup_enabled': True,
            'manual_seed': 0,
            'log_interval': 10,
            'evaluation_interval': 1,
            'early_stopping_patience': 5,
            'early_stopping_delta': 0.001,
            'auto_cluster_interval': 5,
            'cluster_method': 'kmeans',
            'auto_save_enabled': True,
            'auto_save_interval': 5,
            'auto_firing_enabled': False,
            'dream_enabled': True,
            'vram_age_threshold': 300,
            'ram_age_threshold': 600,
            'status_display_interval': 0,
            'neurogenesis_interval': 1,
            'min_cluster_size': 1,
            'prune_frequency': 1,
            'auto_offload': False,
            'benchmark_enabled': False
            , 'model_name': 'marble_default'
            , 'checkpoint_format': 'pickle'
            , 'metrics_history_size': 100
            , 'early_stop_enabled': True
            , 'lobe_sync_interval': 60
            , 'cleanup_batch_size': 500
            , 'remote_sync_enabled': False
            , 'default_activation_function': 'tanh'
            , 'neuron_reservoir_size': 1000
            , 'lobe_decay_rate': 0.98
            , 'super_evolution_mode': False
            , 'dream_decay_arousal_scale': 0.0
            , 'dream_decay_stress_scale': 0.0
        }
        if brain_params is not None:
            brain_defaults.update(brain_params)
        self.brain = Brain(
            self.core,
            self.neuronenblitz,
            self.dataloader,
            remote_client=remote_client,
            torrent_client=torrent_client,
            torrent_map=self.torrent_map,
            metrics_visualizer=self.metrics_visualizer,
            **brain_defaults,
        )
        
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

    # Initialize MARBLE system
    from diffusers import StableDiffusionPipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    marble_system = MARBLE(params, formula=formula, formula_num_neurons=100, 
                          converter_model=pipe.text_encoder, init_from_weights=True)
    core = marble_system.get_core()
    print(f"Core contains {len(core.neurons)} neurons and {len(core.synapses)} synapses.")
    
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
    print(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
    
    # Start background processes
    marble_system.get_brain().start_auto_firing()
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
    norm = (output_scalar - math.floor(output_scalar))
    color_val = int(norm * 255)
    image_array = np.full((128, 128, 3), fill_value=color_val, dtype=np.uint8)
    generated_image = Image.fromarray(image_array)
    generated_image.save("generated_image.png")
    print("Inference completed. The generated image has been saved as 'generated_image.png'.")
