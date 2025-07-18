# YAML Configurable Parameters

This file enumerates every parameter that can be specified in `config.yaml`.
Each entry is listed under its section heading.

## core
- xmin
- xmax
- ymin
- ymax
- width
- height
- max_iter
- representation_size
- message_passing_alpha
- vram_limit_mb
- ram_limit_mb
- disk_limit_mb
- file_tier_path
- init_noise_std
- default_growth_tier

## neuronenblitz
- backtrack_probability
- consolidation_probability
- consolidation_strength
- route_potential_increase
- route_potential_decay
- route_visit_decay_interval
- alternative_connection_prob
- split_probability
- merge_tolerance
- plasticity_threshold
- continue_decay_rate
- struct_weight_multiplier1
- struct_weight_multiplier2
- attention_decay
- max_wander_depth
- learning_rate
- structural_plasticity_enabled
- backtrack_enabled
- loss_scale
- exploration_bonus
- synapse_potential_cap
- attention_update_scale

## brain
- save_threshold
- max_saved_models
- save_dir
- firing_interval_ms
- initial_neurogenesis_factor
- offload_enabled
- torrent_offload_enabled
- mutation_rate
- mutation_strength
- prune_threshold
- dream_num_cycles
- dream_interval
- neurogenesis_base_neurons
- neurogenesis_base_synapses
- offload_threshold
- torrent_offload_threshold
- cluster_high_threshold
- cluster_medium_threshold
- dream_synapse_decay
- neurogenesis_increase_step
- neurogenesis_decrease_step
- max_neurogenesis_factor
- cluster_k
- auto_save_interval
- auto_firing_enabled
- dream_enabled
- vram_age_threshold
- ram_age_threshold
- status_display_interval
- neurogenesis_interval
- min_cluster_size
- prune_frequency
- auto_offload
- benchmark_enabled
- tier_decision_params.vram_usage_threshold
- tier_decision_params.ram_usage_threshold

## meta_controller
- history_length
- adjustment
- min_threshold
- max_threshold

## neuromodulatory_system.initial
- arousal
- stress
- reward
- emotion

## memory_system
- long_term_path
- threshold
- consolidation_interval

## remote_client
- url
- timeout
- max_retries

## torrent_client
- client_id
- buffer_size
- heartbeat_interval

## data_compressor
- compression_level
- compression_enabled

## formula
- formula
- formula_num_neurons
