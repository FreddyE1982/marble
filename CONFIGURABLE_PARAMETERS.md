# YAML Configurable Parameters

This file enumerates every parameter that can be specified in `config.yaml`.
Each entry is listed under its section heading.

## dataset
- source
- num_shards
- shard_index
- offline
- encryption_key
- cache_url: Base URL of ``DatasetCacheServer`` to fetch cached files from.
- use_kuzu_graph
- version_registry
- version
- kuzu_graph.db_path
- kuzu_graph.query
- kuzu_graph.input_column
- kuzu_graph.target_column
- kuzu_graph.limit

## logging
- structured
- log_file

## plugins
- (list of module import paths to load)

## topology_graph
- enabled
- db_path

## pipeline
- async_enabled
- cache_dir
- default_step_memory_limit_mb
- memory_limit_mb (step field limiting memory in MB)
- macro (step field allowing a list of sub-steps)
- tier (step field selecting a remote hardware tier)
- depends_on (step field listing prerequisite step names)
- isolated (step field executing the step in a separate process)

## tool_manager
- enabled
- policy
- tools (mapping of tool identifiers to parameter dictionaries)


## cross_validation
- folds
- seed

## serve_model
- host
- port
## sync
- interval_ms: Interval in milliseconds between cross-device tensor
  synchronisation cycles. Recommended 100–10_000 depending on network speed.

## evolution
- population_size
- selection_size
- generations
- steps_per_candidate
- mutation_rate
- parallelism

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
- weight_init_min
- weight_init_max
- mandelbrot_escape_radius
- mandelbrot_power
- tier_autotune_enabled
- memory_cleanup_interval
- representation_noise_std
- gradient_clip_value
- synapse_weight_decay
- message_passing_iterations
- cluster_algorithm
- vram_limit_mb
- ram_limit_mb
- disk_limit_mb
- file_tier_path
- init_noise_std
- default_growth_tier
- random_seed
- backend
- message_passing_dropout
- synapse_dropout_prob
- synapse_batchnorm_momentum
- representation_activation
- apply_layer_norm
- use_mixed_precision
- quantization_bits
- weight_init_mean
- weight_init_std
- weight_init_type
- weight_init_strategy
- show_message_progress
- message_passing_beta
- attention_temperature
- attention_dropout
- attention_causal
- attention_gating.enabled
- attention_gating.mode
- attention_gating.frequency
- attention_gating.chaos
- salience_weight
- energy_threshold
- reinforcement_learning_enabled
- rl_discount
- rl_learning_rate
- rl_epsilon
- rl_epsilon_decay
- rl_min_epsilon
- early_cleanup_enabled
- pretraining_epochs
- min_cluster_k
- diffusion_steps
- noise_start
- noise_end
- noise_schedule
- workspace_broadcast
- activation_output_dir
- activation_colormap
- memory_system
  - long_term_path
  - threshold
  - consolidation_interval
- cwfl
  - num_basis
  - bandwidth
  - reg_lambda
  - learning_rate
- harmonic
  - base_frequency
  - decay
- fractal
  - target_dimension
- synapse_echo_length
- synapse_echo_decay
- interconnection_prob

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
- weight_decay
- dropout_probability
- dropout_decay_rate
- exploration_decay
- reward_scale
- stress_scale
- remote_fallback
- noise_injection_std
- dynamic_attention_enabled
- backtrack_depth_limit
- synapse_update_cap
- structural_plasticity_enabled
- backtrack_enabled
- loss_scale
- loss_module
- exploration_bonus
- synapse_potential_cap
- attention_update_scale
- plasticity_modulation
- wander_depth_noise
- reward_decay
- synapse_prune_interval
- structural_learning_rate
- remote_timeout
- gradient_noise_std
- min_learning_rate
- max_learning_rate
- top_k_paths
- parallel_wanderers: Number of Neuronenblitz worker threads used for
  ``train_in_parallel`` and parallel wanderers. Default ``1``; values
  below ``1`` are treated as ``1``.
- parallel_update_strategy
- beam_width
- wander_cache_ttl
- wander_anomaly_threshold
- wander_history_size
- phase_rate
- phase_adaptation_rate
- synaptic_fatigue_enabled
- fatigue_increase
- fatigue_decay
- lr_adjustment_factor
- lr_scheduler
- scheduler_steps
- scheduler_gamma
- scheduler.plugin
- epsilon_scheduler
- epsilon_scheduler_steps
- epsilon_scheduler_gamma
- momentum_coefficient
- use_echo_modulation
- reinforcement_learning_enabled
- rl_discount
- rl_epsilon
- rl_epsilon_decay
- rl_min_epsilon
- shortcut_creation_threshold
- chaotic_gating_enabled
- chaotic_gating_param
- chaotic_gate_init
- context_history_size
- context_embedding_decay
- emergent_connection_prob
- concept_association_threshold
- concept_learning_rate
- weight_limit
- wander_cache_size
- rmsprop_beta
- grad_epsilon
- use_experience_replay
- replay_buffer_size
- replay_alpha
- replay_beta
- replay_batch_size
- exploration_entropy_scale
- exploration_entropy_shift
- entropy_epsilon_enabled
- gradient_score_scale
- memory_gate_decay
- memory_gate_strength
- episodic_memory_size
- episodic_memory_threshold
- episodic_memory_prob
- episodic_sim_length
- curiosity_strength
- depth_clip_scaling
- forgetting_rate
- structural_dropout_prob
- gradient_path_score_scale
- use_gradient_path_scoring
- rms_gradient_path_scoring
- activity_gate_exponent
- subpath_cache_size
- subpath_cache_ttl
- monitor_wander_factor
- monitor_epsilon_factor
- use_mixed_precision
- quantization_bits

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
- max_training_epochs
- memory_cleanup_enabled
- manual_seed
- log_interval
- evaluation_interval
- early_stopping_patience
- early_stopping_delta
- auto_cluster_interval
- cluster_method
- auto_save_enabled
- offload_threshold
- torrent_offload_threshold
- cluster_high_threshold
- cluster_medium_threshold
- dream_synapse_decay
- dream_decay_arousal_scale
- dream_decay_stress_scale
- neurogenesis_increase_step
- neurogenesis_decrease_step
- max_neurogenesis_factor
- cluster_k
- auto_save_interval
- backup_enabled
- backup_interval
- backup_dir
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
- benchmark_interval
- tier_decision_params.vram_usage_threshold
- tier_decision_params.ram_usage_threshold
- model_name
- checkpoint_format
- metrics_history_size
- profile_enabled
- profile_log_path
- checkpoint_compress
- profile_interval
- early_stop_enabled
- lobe_sync_interval
- cleanup_batch_size
- remote_sync_enabled
- default_activation_function
- neuron_reservoir_size
- lobe_decay_rate
- dimensional_search.enabled
- dimensional_search.max_size
- dimensional_search.improvement_threshold
- dimensional_search.plateau_epochs

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

## hybrid_memory
- vector_store_path
- symbolic_store_path
- kuzu_store_path
- max_entries


## network.remote_client
- url
- timeout
- max_retries
- backoff_factor
- track_latency
- auth_token
- ssl_verify
- connect_retry_interval
- heartbeat_timeout
- use_compression

## network.torrent_client
- client_id
- buffer_size
- heartbeat_interval

## data_compressor
- compression_level
- compression_enabled
- delta_encoding
- compression_algorithm
- quantization_bits
- sparse_threshold
## remote_hardware
- tier_plugin: Import path of a module exposing ``get_remote_tier`` used to
  instantiate a custom remote hardware tier.
- grpc.address: Host and port for the default ``GrpcRemoteTier``
  implementation.
- grpc.max_retries: Number of times to retry gRPC calls on failure.
- grpc.backoff_factor: Multiplier for exponential backoff between retries.

## dataloader
- tensor_dtype
- track_metadata
- enable_round_trip_check
- round_trip_penalty
- tokenizer_type
- tokenizer_json
- tokenizer_vocab_size

## experiment_tracker
- enabled
- project
- entity
- run_name

## formula
- formula
- formula_num_neurons

## network.remote_server
- enabled
- host
- port
- remote_url
- auth_token
- ssl_enabled
- ssl_cert_file
- ssl_key_file
- max_connections
- compression_level
- compression_enabled
## remote_hardware
- tier_plugin: Module path implementing the remote hardware API.
- grpc.address: gRPC service location when using ``GrpcRemoteTier``.

## metrics_visualizer
- fig_width
- fig_height
- refresh_rate
- color_scheme
- show_neuron_ids
- dpi
- track_memory_usage
- track_cpu_usage
- log_dir
- csv_log_path
- json_log_path
- anomaly_std_threshold
## metrics_dashboard
- enabled
- host
- port
- update_interval
- window_size

## lobe_manager
- attention_increase_factor
- attention_decrease_factor

## brain (additional)
- loss_growth_threshold
- auto_neurogenesis_prob
- dream_cycle_sleep
- dream_replay_buffer_size
- dream_replay_batch_size
- dream_replay_weighting: Sampling strategy for the dream replay buffer
  ("linear", "exponential", "quadratic", "sqrt", "uniform").
- dream_instant_buffer_size
- dream_housekeeping_threshold: Minimum salience (0-1) below which experiences
  are pruned during dream housekeeping
- super_evolution_mode

## autograd
- enabled
- learning_rate
- gradient_accumulation_steps
## global_workspace
- enabled
- capacity
## attention_codelets
- enabled
- coalition_size
## pytorch_challenge
- enabled
- loss_penalty
- speed_penalty
- size_penalty

## gpt
- enabled
- vocab_size
- block_size
- num_layers
- num_heads
- hidden_dim
- learning_rate
- num_train_steps
- dataset_path
- batch_size

## dataset
- num_shards
- shard_index
- offline
- encryption_key
- source
- use_kuzu_graph
- kuzu_graph.db_path
- kuzu_graph.query
- kuzu_graph.input_column
- kuzu_graph.target_column
- kuzu_graph.limit

## distillation
- enabled
- alpha
## logging
- structured
- log_file

- teacher_model

## reinforcement_learning
- enabled
- algorithm
- episodes
- max_steps
- discount_factor
- epsilon_start
- epsilon_decay
- epsilon_min
- seed
- double_q

## contrastive_learning
- enabled
- temperature
- epochs
- batch_size

## hebbian_learning
- learning_rate
- weight_decay

## adversarial_learning
- enabled
- epochs
- batch_size
- noise_dim

## autoencoder_learning
- enabled
- epochs
- batch_size
- noise_std
- noise_decay

## semi_supervised_learning
- enabled
- epochs
- batch_size
- unlabeled_weight
## federated_learning
- enabled
- rounds
- local_epochs

## curriculum_learning
- enabled
- epochs
- schedule
## meta_learning
- enabled
- epochs
- inner_steps
- meta_lr
- distill_alpha
## transfer_learning
- enabled
- epochs
- batch_size
- freeze_fraction

## continual_learning
- enabled
- epochs
- memory_size

## imitation_learning
- enabled
- epochs
- max_history

## harmonic_resonance_learning
- enabled
- epochs
- base_frequency
- decay

## synaptic_echo_learning
- enabled
- epochs
- echo_influence

## fractal_dimension_learning
- enabled
- epochs
- target_dimension

## quantum_flux_learning
- enabled
- epochs
- phase_rate

## dream_reinforcement_learning
- enabled
- episodes
- dream_cycles
- dream_strength
- dream_interval
- dream_cycle_duration

## continuous_weight_field_learning
- enabled
- epochs
- num_basis
- bandwidth
- reg_lambda
- learning_rate

## neural_schema_induction
- enabled
- epochs
- support_threshold
- max_schema_size

## conceptual_integration
- enabled
- blend_probability
- similarity_threshold

## n_dimensional_topology
- enabled
- target_dimensions
- attention_threshold
- loss_improve_threshold
- stagnation_epochs

## unified_learning
- enabled
- gating_hidden
- log_path

## theory_of_mind
- hidden_size
- num_layers
- prediction_horizon
- memory_slots
- attention_hops
- mismatch_threshold

## predictive_coding
- num_layers
- latent_dim
- learning_rate

## experiments
- name
- core
- neuronenblitz
