# MARBLE Redevelopment Plan

This document enumerates every step required to rebuild MARBLE from scratch with full feature and algorithmic parity. All modules must be reimplemented without simplification and every configuration key must be exercised. No GUI components are to be created.

## 1. Repository Bootstrap
### 1.1 Initialize new Git repository
- Create repository structure.
- Configure pre-commit, linting, formatting, type checking.

### 1.2 Dependency and environment setup
- Define pyproject.toml and requirements.txt mirroring current repo.
- Implement Python virtual environment setup scripts.
- Provide GPU/CPU detection utilities.
- Continuous integration scripts for linting, tests and publishing.

## 2. Configuration and Parameter Management
### 2.1 YAML configuration system
- Implement config schema, loader, generator and synchronization service.
- Provide yaml-manual explaining every parameter and its allowed range.

### 2.2 Parameter integration
- Enumerate every key from config.yaml and ensure each is mapped to real code paths.
- Implement validation ensuring no unused keys.
- Add unit tests verifying that every parameter is exercised at least once.

## 3. Core Architecture
### 3.1 Event and message infrastructure
- Implement event_bus and message_bus with asynchronous queues.
- Provide publish/subscribe API and serialization hooks.

### 3.2 Core neural substrate
- Recreate marble_core with Neuron, Synapse, and perform_message_passing.
- Include structural plasticity operations, neuron/synapse type registries and weight limiting.
- Implement MarbleBrain, MarbleLobes, MarbleGraphBuilder and GraphCache for topology management.

### 3.3 Memory systems
- Implement memory_pool, memory_manager, episodic memory, hybrid memory, prompt memory and Kuzu-backed tiers.
- Include forgetfulness and consolidation algorithms.

### 3.4 Plugin system
- Recreate plugin_system with dynamic loading and registration of neuron, synapse and loss modules.
- Implement pipeline, scheduler, tool and learning plugin registries.

### 3.5 Dataset infrastructure
- Implement dataset loader, replication, watcher, streaming datasets, encryption and versioning.
- Provide dataset cache server and history CLI.

## 4. Neuronenblitz Algorithm
### 4.1 Overview
Neuronenblitz is MARBLE's core adaptive exploration and learning mechanism. It performs stochastic wandering over the neural graph while adjusting synaptic weights and structure.

### 4.2 Data structures
- State objects tracking route potentials, synapse fatigue, context history, wander cache and subpath cache.
- Dynamic attention span module controlling focus across neuron types.

### 4.3 Dynamic wander algorithm
1. Start from a given neuron; set depth d = 0.
2. At each step choose among outgoing synapses using potential scores p_s and exploration noise:
   P(s_i) = exp(p_s_i / tau) / sum_j exp(p_s_j / tau).
3. Update route potential for traversed synapse:
   p' = p * route_potential_decay + route_potential_increase.
4. Track depth and terminate when d >= max_wander_depth or target reached.
5. Optionally backtrack with probability backtrack_probability up to backtrack_depth_limit.

### 4.4 Output computation
- Each neuron's output is computed by applying combine_fn over inputs x and weights w:
  y = max(x * w, 0) (default).
- Loss is computed as ell = loss_fn(t, y) or via loss_module.

### 4.5 Validation and learning
1. Compute validation scale v = validation_fn(t, y).
2. Error e = v * ell.
3. Weight update contribution for a synapse on path length L:
   delta_w = learning_rate * weight_update_fn(s, e, L).
   Default delta_w = learning_rate * (e * s) / (L + 1).
4. Apply dropout with probability dropout_probability (decayed per step).
5. Update weight with decay and clipping:
   w' = clip(w + delta_w - weight_decay * w, -weight_limit, weight_limit).

### 4.6 Structural plasticity
- For each visited synapse, with probability split_probability create alternative connection; with probability consolidation_probability increase weight by consolidation_strength.
- Merge similar synapses if weight difference < merge_tolerance.
- Prune synapses every synapse_prune_interval steps when potential < synapse_potential_cap.

### 4.7 Attention and fatigue
- Update type-specific attention:
  a' = a * attention_decay + attention_update_scale * activity.
- Synaptic fatigue:
  f' = f * fatigue_decay + fatigue_increase * activity.
- Attention span threshold controls dynamic span module.

### 4.8 Reinforcement learning integration
- Q-learning weight update for state-action pair (s,a):
  Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).
- Epsilon scheduling:
  eps_{t+1} = max(rl_min_epsilon, eps_t * rl_epsilon_decay).
- Discounted return for policy gradients:
  G_t = sum_{k=0}^inf gamma^k * r_{t+k+1}.

### 4.9 Optimization and scheduling
- RMSProp accumulator for gradient g:
  v' = beta * v + (1-beta) * g^2;
  g_adj = g / sqrt(v' + grad_epsilon).
- Learning rate scheduler:
  lr_{t+1} = clip(lr_t * scheduler_gamma, min_learning_rate, max_learning_rate).
- Epsilon scheduler analogous to learning rate scheduler.

### 4.10 Chaotic gating and phase adaptation
- Chaotic gate using logistic map:
  c_{t+1} = chaotic_gating_param * c_t * (1 - c_t).
- Phase update:
  phi_{t+1} = phi_t + phase_rate + phase_adaptation_rate * e.

### 4.11 Experience replay and memory gating
- Prioritized replay probability:
  P_i = (p_i^replay_alpha) / sum_j (p_j^replay_alpha).
- Importance weight:
  w_i = (1 / (N * P_i))^replay_beta.
- Memory gate value:
  g' = g * memory_gate_decay + memory_gate_strength * abs(e).

### 4.12 Parameter inventory
All parameters of Neuronenblitz must be exposed and exercised:
backtrack_probability, consolidation_probability, consolidation_strength, route_potential_increase, route_potential_decay, route_visit_decay_interval, alternative_connection_prob, split_probability, merge_tolerance, combine_fn, loss_fn, loss_module, weight_update_fn, validation_fn, plasticity_threshold, continue_decay_rate, struct_weight_multiplier1, struct_weight_multiplier2, attention_decay, max_wander_depth, learning_rate, weight_decay, dropout_probability, dropout_decay_rate, exploration_decay, reward_scale, stress_scale, auto_update, dataset_path, remote_fallback, noise_injection_std, dynamic_attention_enabled, backtrack_depth_limit, synapse_update_cap, structural_plasticity_enabled, backtrack_enabled, loss_scale, exploration_bonus, synapse_potential_cap, attention_update_scale, attention_span_threshold, max_attention_span, span_module, plasticity_modulation, wander_depth_noise, reward_decay, synapse_prune_interval, gradient_prune_ratio, structural_learning_rate, remote_timeout, gradient_noise_std, min_learning_rate, max_learning_rate, top_k_paths, parallel_wanderers, parallel_update_strategy, beam_width, synaptic_fatigue_enabled, fatigue_increase, fatigue_decay, lr_adjustment_factor, lr_scheduler, scheduler_steps, scheduler_gamma, epsilon_scheduler, epsilon_scheduler_steps, epsilon_scheduler_gamma, momentum_coefficient, reinforcement_learning_enabled, rl_discount, rl_epsilon, rl_epsilon_decay, rl_min_epsilon, entropy_epsilon_enabled, shortcut_creation_threshold, use_echo_modulation, wander_cache_ttl, phase_rate, phase_adaptation_rate, chaotic_gating_enabled, chaotic_gating_param, chaotic_gate_init, context_history_size, context_embedding_decay, emergent_connection_prob, concept_association_threshold, concept_learning_rate, weight_limit, wander_cache_size, plasticity_history_size, rmsprop_beta, grad_epsilon, use_experience_replay, replay_buffer_size, replay_alpha, replay_beta, replay_batch_size, exploration_entropy_scale, exploration_entropy_shift, gradient_score_scale, memory_gate_decay, memory_gate_strength, episodic_memory_size, episodic_memory_threshold, episodic_memory_prob, curiosity_strength, depth_clip_scaling, forgetting_rate, structural_dropout_prob, gradient_path_score_scale, use_gradient_path_scoring, rms_gradient_path_scoring, activity_gate_exponent, subpath_cache_size, gradient_accumulation_steps, wander_anomaly_threshold, wander_history_size, subpath_cache_ttl, monitor_wander_factor, monitor_epsilon_factor, episodic_sim_length, use_mixed_precision, remote_client, torrent_client, torrent_map, metrics_visualizer.

## 5. Learning Modules
For each learning paradigm below, reimplement training loops, loss functions, evaluation metrics and configuration hooks.

### 5.1 Contrastive Learning
- Implement InfoNCE loss:
  L = -(1/N) * sum_{i=1}^N log( exp(sim(z_i, z_i_plus)/tau) / sum_{j=1}^{2N} exp(sim(z_i, z_j)/tau) ).
- Integrate with ContrastiveLearner and pipeline.

### 5.2 Imitation Learning
- Behavioural cloning using cross-entropy loss.
- Integrate dataset loader and ImitationLearner.

### 5.3 Autoencoder Learning
- Reconstruction loss L = ||x - x_hat||_2^2.
- Optional KL divergence regularization for variational variants.

### 5.4 Reinforcement Learning
- Implement Q-learning agents, policy gradients, SAC baseline and hierarchical RL.
- Provide replay buffer, exploration strategies and reward scaling.

### 5.5 Curriculum and Transfer Learning
- Implement schedulers for task difficulty and transfer pipelines.
- Gradually adjust task distribution D_t to D_{t+1} such that KL(D_{t+1} || D_t) < epsilon.

### 5.6 Meta and Continual Learning
- Implement meta-parameter controller adjusting plasticity using validation losses.
- Continual learning with elastic weight consolidation: penalty sum_i (lambda/2) * F_i * (theta_i - theta_i_star)^2.

### 5.7 Adversarial and Fractal Learners
- GAN objective: min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))].
- Fractal dimension learning using correlation dimension D2 = lim_{r->0} (log C(r) / log r).

### 5.8 Harmonic Resonance and Quantum Flux
- Harmonic resonance loss using Fourier transforms to match frequency spectra.
- Quantum flux learning uses complex-valued amplitudes with unitary constraint U^dagger U = I.

## 6. Pipelines and Orchestration
### 6.1 Pipeline framework
- Implement pipeline, pipeline_cli, pipeline_schema, highlevel_pipeline and examples.
- Provide step registration, DAG validation and execution engine.

### 6.2 Scheduler plugins
- Implement plugin interface for custom schedulers such as dream_scheduler and remote_worker_pool.
- Ensure each scheduler option in config is wired to executable code.

### 6.3 Tool and learner plugins
- Recreate tool_manager_plugin, tool_plugins and learning_plugins with dynamic discovery.

## 7. Memory and Simulation Systems
### 7.1 Episodic simulation and dream modules
- Implement episodic_simulation, dream_replay_buffer, dream_scheduler and reinforcement consolidation.

### 7.2 Prompt and attention codelets
- Implement prompt_memory, attention_codelets, attention_utils for managing context and focus.

### 7.3 Self-monitoring and metrics
- Integrate self_monitoring and metrics_dashboard to track errors, wander anomalies and plasticity history.

## 8. Utilities and Interop
### 8.1 External framework interop
- Implement PyTorch and TensorFlow interop layers (pytorch_to_marble, torch_interop, tensorflow_interop).
- Provide model import/export (convert_model, marble_to_pytorch, torch_model_io).

### 8.2 Remote and distributed execution
- Implement remote_offload, remote_worker_pool, distributed_training and torrent-based model exchange.
- Include networkx graph export, web API and database query tools.

### 8.3 Experiment tracking and logging
- Integrate experiment_tracker, logging_utils and usage_profiler with configurable backends.

## 9. Testing and Validation
### 9.1 Unit tests
- Write pytest suites for every module and parameter combination.
- Ensure tests cover CPU and GPU execution paths.

### 9.2 Integration tests
- Simulate end-to-end pipelines verifying data flow from datasets through learners and Neuronenblitz.

### 9.3 Performance and stress tests
- Recreate benchmarks (benchmark_* scripts) to verify parity.

### 9.4 Config coverage
- Add tests asserting no orphaned configuration keys.

## 10. Documentation and Tutorials
- Regenerate README, ARCHITECTURE_OVERVIEW, ML_PARADIGMS_HANDBOOK and a new multi-project TUTORIAL without GUI references.
- Maintain ROADMAP, TROUBLESHOOTING, HIGHLEVEL_PIPELINE_TUTORIAL and configuration manuals.

## 11. Release and Maintenance
- Provide versioning strategy, changelog generation and automated publishing.
- Establish code-style guidelines and contribution templates.

Following this sequence will rebuild MARBLE with complete feature parity and precise parameter utilization without introducing a GUI.

