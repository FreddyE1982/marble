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


### 2.3 Configuration-driven module instantiation
- Provide `DotDict` utility for attribute-style access to nested configuration dictionaries, ensuring recursive conversion and update semantics.
- Implement `config_loader` that materializes every configuration section into runtime objects.
- Structured logging options: enable/disable structured format, log file path, level, message format, datefmt, propagation, log rotation with `max_bytes`, `backup_count` and `encoding`.
- JSONFormatter writes logs as JSON with optional rotating file handler.
- Scheduler selection via `configure_scheduler` and plugin directories loaded through `load_plugins`.
- Instantiate MetaParameterController (history_length, adjustment, min_threshold, max_threshold) and NeuromodulatorySystem with initial context values.
- Build MemorySystem with `long_term_path`, consolidation `threshold`, `consolidation_interval` and optional hybrid memory parameters.
- DataCompressor settings: `compression_level`, `compression_enabled`, `sparse_threshold`, `quantization_bits` and optional delta encoding.
- DataLoader parameters: tensor dtype, metadata tracking, automatic
  encode/decode/tokenization with round-trip verification/penalty and
  - tokenizer type/json/vocab_size.
- DataLoader parameters: tensor dtype, metadata tracking, automatic encode/decode/tokenization with round-trip verification/penalty and tokenizer type/json/vocab_size.
- Autograd layer configuration with `autograd_params` (`enabled`, `learning_rate`, `gradient_accumulation_steps`, optional `scheduler`) controlling gradient accumulation and learning-rate scheduling.
- Autoencoder learning section with `enabled`, `epochs`, `batch_size`, `noise_std`,
  and `noise_decay` controls default denoising behaviour.
- `core.salience_weight` scales attention proposal saliences when forming coalitions.
- Network components: `RemoteBrainClient`, `RemoteBrainServer`, torrent client/tracker and remote tier plugins.
- Metrics visualizer and optional Kùzu experiment/topology trackers.
- Optional modules activated through config: predictive_coding, tool_manager, tensor_sync_service, unified_learning, global_workspace, attention_codelets, conceptual_integration, theory_of_mind and weight quantization. `predictive_coding.activate(num_layers, latent_dim, learning_rate)` returns an object whose repeated `step(x)` calls should monotonically reduce squared error.
- Conditional training helpers: advanced GPT training, reinforcement learning (policy gradient or Q-learning), dream reinforcement, adversarial, transfer, semi‑supervised, federated, curriculum, imitation, harmonic resonance, quantum flux, synaptic echo and fractal dimension learners.
- UnifiedLearner coordinates multiple paradigms via a gating network; UnifiedPairsPipeline and TransferPairsPipeline feed pair datasets.
- OmniLearner sequences all learners including continuous weight field, neural schema induction and conceptual integration in one step.
## 3. Core Architecture
### 3.1 Event and message infrastructure
- Implement event_bus and message_bus with asynchronous queues.
- MessageBus supports direct, broadcast and reply communication, logs history for NetworkX influence graphs and offers `AsyncDispatcher` background delivery with configurable poll_interval.
- Provide publish/subscribe API and serialization hooks.
- EventBus supports event filtering, rate limiting and unified `ProgressEvent` schema.

### 3.2 Core neural substrate
- Include `GraphCache` for torch.jit precompilation keyed by tensor shape/dtype, streaming utilities `stream_graph_chunks` and `identify_memory_hotspots`, and `graph_viz.sankey_figure` for interactive topology exploration.
- Recreate marble_core with Neuron, Synapse, and perform_message_passing.
- Include structural plasticity operations, neuron/synapse type registries and weight limiting.
- Implement MarbleBrain, MarbleLobes, MarbleGraphBuilder and GraphCache for topology management.
- Implement tiered memory system using `TierMeta` registry with default tiers: `VramTier`, `RamTier`, `DiskTier`, `FileTier` (writes modified data to disk) and `RemoteTier` for HTTP offload.
- Neuron parameter validation must enforce positive stride, dropout probability \(0\le p\le 1\), appropriate kernel dimensionality, non-negative padding/output_padding, non-negative `negative_slope`, positive `alpha`, and momentum \(0<m<1\).


- Initial neuron representations seeded via Mandelbrot fractals with optional Gaussian noise.
- Message passing employs `AttentionModule` with temperature scaling, sine or chaotic gating, dropout and mixed-precision layer-normalised MLP updates.
- `DynamicSpanModule` applies cumulative-softmax masking with configurable `threshold` and `max_span` to cap traversal length and integrates into Neuronenblitz `dynamic_wander`.
- `attention_codelets` let codelets emit `AttentionProposal(score, content)`; `form_coalition` selects top proposals using optional salience scores weighted by `core.salience_weight`, `broadcast_coalition` forwards winners to `global_workspace`, and workspace gating adjusts future scores based on published events.
- `interconnect_cores` merges multiple cores and optionally creates cross-core synapses based on interconnection probability.
- Ensure backend equivalence where Mandelbrot seed generation and message passing yield identical results on NumPy and JAX backends.
### 3.3 Memory systems
- Implement `HybridMemory` combining vector similarity search, symbolic key-value store and optional Kùzu-backed tier with temporal forgetting.
- Implement memory_pool, memory_manager, episodic memory, hybrid memory, prompt memory and Kuzu-backed tiers.
  - `PromptMemory` caches input/output pairs, builds composite prompts within a character limit and supports JSON `serialize`/`load`. It evicts oldest pairs FIFO, preserves timestamps during serialization, enforces `max_chars` when composing, inserts 5000 entries in under 10ms each and maintains comparable CPU/GPU insertion performance.
- Implement KuzuMemoryTier using KùzuGraphDatabase with MERGE-based inserts, cosine-similarity query over stored vectors and timestamp-driven `forget_old` trimming.
- Include forgetfulness and consolidation algorithms.

### 3.4 Plugin system
- Recreate plugin_system with dynamic loading and registration of neuron, synapse and loss modules.
- Implement pipeline, scheduler, tool and learning plugin registries.
- Provide pipeline execution engine:
  - Add steps with `add_step`, resolve dependencies and detect cycles across CPU/GPU devices.
  - Support macro steps via `add_macro` and `rollback` to remove cached future results on errors.
  - Emit progress events through `global_event_bus` using `PROGRESS_EVENT`.
  - Allow pre and post hooks that move tensors to the active device and clean GPU memory.
  - Validate steps against JSON schemas; `diff_config` highlights changes and `dataset_summaries` reports dataset pair counts.
  - `execute` accepts `log_callback` for streaming log lines.
  - Load `PipelinePlugin` implementations from directories, initialising with devices and tearing down after execution.
  - Provide graph builders `pipeline_to_networkx` and `pipeline_to_core` verifying directed acyclic graphs with matching node counts.
- Provide LearningModule base with `register_learning_module`, `get_learning_module` and `load_learning_plugins` discovering entry points or plugin directories.
- Build ToolPlugin base and ToolManagerPlugin with heuristic selection policy and optional MCP/MessageBus integration.

- Provide `register_neuron_type`, `register_synapse_type` and `register_loss_module` to extend `marble_core` type registries.
- Implement `load_plugins` scanning directories for modules defining `register` with neuron/synapse and optional loss callbacks.
### 3.5 Dataset infrastructure
#### 3.5.1 Automatic encoding and tokenization
- Implement `DataLoader` with plugin registries for custom encoders/decoders,
  compression via `DataCompressor`, optional tokenizers and full metadata
   tracking.
- For unregistered types, pickle the object and store its module and class metadata so arbitrary Python objects can be restored.
- Encoding transforms an object \(o\) into bytes
  \(b = \text{compressor.compress}(\text{pickle}(o))\), converts to tensors via
  `np.frombuffer`, and decodes by inverting these steps.
- Round-trip verification applies penalty when
  \(\text{decode} (\text{encode}(o)) \ne o\).
- If a tokenizer is supplied and the value is text, tokenise using
  `tokenizer.encode` before compression and decode with `tokenizer.decode`.

#### 3.5.2 Bit-level conversion utilities
- Provide symmetric conversions `object_to_bytes`/`bytes_to_object` with
  optional AES-256-GCM encryption and serializer selection
  (pickle/json/msgpack).
- Implement `bytes_to_tensors` converting bytes to \((n,8)\) bit tensors via bit
  masks and `tensors_to_bytes` reversing the process with weighted sums.
- Utilities `flatten_tensor_to_bitstream` and `unflatten_bitstream_to_tensor`
  enable vocabulary mining.

#### 3.5.3 BitTensorDataset
- Implement `DatasetPair` container and `augment_bit_tensor` for (x' = x \oplus f_p + n_p).
- `build_vocab` counts bit patterns and assigns token IDs starting at 256; encoding uses `encode_with_vocab` and decoding uses `decode_with_vocab`.
- Vocabulary controls: `use_vocab`, `max_vocab_size`, `min_occurrence`, `max_word_length`, configurable `start_id`, reuse or persist vocab files and `adapt_vocab`/`rebuild_vocab` when appending pairs.
- Dataset features include optional compression, AES-GCM encryption with required key, SHA256 indexing, history snapshots, deterministic `split`/`merge`, `split_deterministic` with salt, `shuffle`, `hash`/`hash_pair` lookup and Annoy-based `build_ann_index`/`nearest_neighbors`.
- Pair operations support `add_pair`, `extend`, `append_pairs`, synchronous or asynchronous `add_stream_pair`, `map_pairs`, `filter_pairs`, `deduplicate`, `patch_pairs`, `prune_invalid`, `augment_bits` and `release_memory`.
- Persistence and iteration provide `save`/`load` (memory-mapped or cached), `save_async`, `to_json`/`from_json`, `save_vocab`/`load_vocab`, dataset `summary`, `iter_decoded`, `collate_fn` batching and checksum verification on load.
#### 3.5.4 Streaming datasets
- Implement `BitTensorStreamingDataset` providing `seek_to`, `seek_forward` and `seek_backward` to navigate underlying streams.
- `get_virtual_batch` supports cached or streaming retrieval with configurable `virtual_batch_size` and re-access of previously fetched batches.
- Streaming remains lazy: accessing items triggers selection tracking in datasets supporting `select`, `skip` or `take` interfaces.
- Integrate HuggingFace `IterableDataset` and custom generators with on-the-fly encoding to tensors.
- Implement `StreamingDatasetStep` asynchronously prefetching batches to CPU/GPU via an `asyncio.Queue` and yielding `{"inputs": tensor, "targets": tensor}` dictionaries.

#### 3.5.5 Dataset loaders and preprocessing
- `load_dataset` handles URL caching, prefetch threads, AES-encrypted downloads,
  sharding, dependency tracking, filter expressions and distributed sharding;
  all values pass through `DataLoader.encode`/`decode` enabling arbitrary
  binary or textual data.
- `prefetch_dataset` spawns daemon threads and `wait_for_prefetch` acts as a
  barrier.
- Provide `StreamingCSVLoader` with resumable offsets and per-line
  tokenisation via `tokenize_line`, plus `export_dataset` and
  `clear_dataset_cache` utilities.
- Support `load_kuzu_graph` for Cypher queries and
  `load_training_data_from_config` that forwards configuration keys and applies
  dataset versioning.
- `RemoteWorkerPool` executes preprocessing functions in daemon processes via
  pipes, retries failed jobs and restarts dead workers.
- `PreprocessingPipeline` caches step outputs by `dataset_id`, applies sequential transformations and reuses cache on repeated calls; optional `DataLoader` tokenizes text datasets.

#### 3.5.6 Dataset management tools
- Include dataset cache server, history CLI, watcher, versioning CLI and
  replication/synchronisation services.
- Dataset encryption helpers offer AES-256-GCM key generation and
  `encrypt_tensor`/`decrypt_tensor` for tensors plus
  `encrypt_bytes`/`decrypt_bytes` for arbitrary data.
- Provide `KuzuGraphDatabase` wrapper with context-managed connections and CRUD
  helpers; `TopologyKuzuTracker` mirrors neural topology to Kùzu.

### 3.6 Brain coordination and neurogenesis
- Add `GoalManager` maintaining goal hierarchies and reward shaping; integrate global workspace via `BroadcastMessage` queue for cross-module signalling.
- Implement `MarbleBrain` to supervise Neuronenblitz and global learning state.
- Add neuromodulatory system providing context values (arousal, stress, reward, emotion).
- MetaParameterController adjusts plasticity threshold based on validation loss trends.
- NDimensionalTopologyManager dynamically increases or decreases representation size driven by attention and loss stagnation.
- LobeManager groups neurons into lobes and performs self-attention based on cluster IDs.
- Brain training runs pretraining for `pretraining_epochs` once before standard epochs, enforces `min_cluster_k` during neuron clustering and triggers clustering every `auto_cluster_interval` epochs.
- Implement neurogenesis controller:
  1. Compute growth factor
     \(f = (1+\max(\text{arousal},\text{reward})) \cdot \text{neurogenesis\_factor}\).
  2. Create \(N = \lfloor \text{base\_neurons} \cdot f \rfloor\) neurons and
     \(S = \lfloor \text{base\_synapses} \cdot f \rfloor\) synapses.
  3. Choose neuron types via Neuronenblitz preferred or combined attention.
  4. Invoke `core.expand` with \(N,S\) and selected types.
- Implement adaptive factor update:
  \(
  \text{neurogenesis\_factor} \leftarrow
  \min(\text{max\_factor}, \text{factor} + \text{increase\_step})
  \) when validation loss rises and
  \(
  \text{neurogenesis\_factor} \leftarrow
  \max(1.0, \text{factor} - \text{decrease\_step})
  \) when it drops.
- Add autonomous trigger:
  \(P(\text{growth}) = \text{auto\_neurogenesis\_prob} \cdot \min(1, \text{val\_loss})\).
- Implement resource-aware tier selection `choose_growth_tier`:
  1. If VRAM usage ≥ `vram_limit * vram_usage_threshold` or average VRAM neuron
     age > 300 s then
     1. If RAM usage ≥ `ram_limit * ram_usage_threshold` or average RAM neuron
        age > 600 s choose tier `file` (fallback `disk`).
     2. Else choose tier `ram`.
  2. Otherwise pick configured `default_growth_tier` (falling back to `vram`).
- Compute dream-induced decay
  \(d = \text{dream\_synapse\_decay} (1 + s_a \cdot \text{arousal})(1 - s_s \cdot \text{stress})\).
- Expose `maybe_autonomous_neurogenesis` triggering growth with probability
  \(p = \text{auto\_neurogenesis\_prob} \cdot \min(1, \text{val\_loss})\).
### 3.7 Theory of Mind module
- Define `ToMInput` dataclass carrying agent id, character id, observation
  sequences and belief-state tensors with validation helpers.
- Implement `ToMModule` belief memory with key and value slots and multi-hop
  attention \(q_{t+1}=q_t + \text{softmax}(K q_t)V\).
- Provide `CharacterModel` LSTM and `TheoryOfMind` container managing
  per-character models and mismatch records.
- Observation processing:
  1. Write beliefs to memory.
  2. Retrieve beliefs, compute mismatch \(m = \mathrm{mean}[(r-v)^2]\) and
     store cases above `mismatch_threshold`.
  3. Predict next observation and log error
     \(e = \mathrm{mean}[(\hat{o} - o_{-1})^2]\) to Neuronenblitz.
  4. Publish predictions through `global_workspace`.
- Provide `get_mismatches`, JSON `save_mismatches`, forward `predict` and an
  `activate` helper attaching the module to MARBLE.
### 3.8 Training orchestration:
  - Each epoch: call `update_neurogenesis_factor(val_loss)` then
    `maybe_autonomous_neurogenesis(val_loss)` to adapt growth.
  - Optional pretraining epochs using identity targets.
  - Graph precompilation for CUDA kernels via `precompile_simple_mlp`.
  - Epoch loop with tqdm progress, validation hooks and early stopping when
    validation loss fails to improve by `early_stopping_delta` for
    `early_stopping_patience` epochs.
  - Metrics visualizer logging loss, VRAM/RAM/GPU usage, neuromodulatory
    context and meta-controller history.
  - Dream-induced decay through `dream_scheduler.replay` and periodic
    `dream`/`start_dreaming` threads; `compute_dream_decay` scales synapse
    weights during sleep.
  - Auto-firing mode generating random inputs at interval
    `firing_interval_ms` in a background thread.
  - DimensionalitySearch expands representation size when validation loss plateaus.
  - Mutation and pruning utilities `mutate_synapses` and `prune_weak_synapses`
    with evolutionary wrapper `evolve`.
  - High-attention offloading to remote or torrent clients and memory cleanup
    migrating aged neurons across VRAM→RAM→disk tiers.
  - Asynchronous `start_training`/`train_async` threads and `wait_for_training`
    synchronization.


### 3.7 Diffusion-based generation
- Implement `DiffusionCore` with configurable diffusion_steps and linear or cosine noise schedules.
- Support hybrid memory retrieval, schema induction, predictive coding, continuous weight field learning, harmonic resonance and fractal dimension learners.
- Provide optional workspace broadcast, remote offloading when VRAM exceeds thresholds and activation heatmap logging.
- Train via `DiffusionPairsPipeline` wrapping `DiffusionCore` with `DataLoader`, `Brain` and `Neuronenblitz` components.
## 4. Neuronenblitz Algorithm
### 4.1 Overview
Neuronenblitz is MARBLE's core adaptive exploration and learning mechanism. It performs stochastic wandering over the neural graph while adjusting synaptic weights and structure.

### 4.2 Data structures
- State objects tracking route potentials, synapse fatigue, context history, wander cache and subpath cache.
- Dynamic attention span module controlling focus across neuron types.

### 4.3 Dynamic wander algorithm
1. Start from a given neuron; set depth d = 0.
2. At each step choose among outgoing synapses using potential scores p_s and exploration noise:
   P(s_i) = e^{p_{s_i}/\tau} / \sum_j e^{p_{s_j}/\tau}.
3. Update route potential for traversed synapse:
   p' = \min(cap, p * route_potential_decay + route_potential_increase).
4. Track depth and terminate when d >= max_wander_depth or target reached.
5. With probability backtrack_probability and depth < backtrack_depth_limit, backtrack one step and retry.
6. Apply dropout by removing each candidate synapse with probability dropout_probability.
7. If target neuron resides on a remote tier, invoke `remote_client.process` with
   timeout; optionally fall back to local processing or torrent shard via
   `torrent_client`.
8. Bias initial path via episodic memory replay for up to
   `episodic_sim_length` steps.
9. Optionally perform beam search of width b to keep top-b paths scored by
   cumulative potential and novelty.
10. Cache explored subpaths with resulting neuron values in a wander cache with
    TTL and LRU eviction; reuse cached paths when `apply_plasticity` is False.
11. After each wander decay route potentials and prune low-potential synapses at
    configured intervals.

### 4.4 Output computation
- Each neuron's output is computed by applying combine_fn over inputs x and weights w:
  y = max(x * w, 0) (default).
- Loss is computed as ell = loss_fn(t, y) or via loss_module.

### 4.5 Validation and learning
1. Modulate plasticity based on neuromodulators: \(reward = ctx_{reward}\cdot reward\_scale\), \(stress = ctx_{stress}\cdot stress\_scale\), \(plasticity\_threshold = \max(0.5, plasticity\_threshold - (reward - stress)\cdot plasticity\_modulation)\). Append enriched context (markers, goals, tom) to `context_history` and decay reward/stress by `reward_decay`.
2. Compute validation scale v = validation_fn(t, y).
3. Error e = v * ell.
4. For each synapse with source value s and path length L:
   - Raw gradient Δ = weight_update_fn(s, e, L) (default Δ = (e·s)/(L+1)).
   - Eligibility-modulated gradient Δ' = Δ * eligibility_trace.
   - v' = β v_prev + (1-β)Δ'^2
   - scaled = Δ'/√(v'+ε)
   - m' = μ m_prev + scaled
   - update = lr*(μ m' + scaled)
   - phase_factor = cos(φ_syn - φ_global); if chaotic gating enabled multiply by g(t)
   - fatigue gate = max(0, 1 - fatigue)
   - activity gate = 1/(1 + visit_count^{activity_gate_exponent})
   - update *= phase_factor * fatigue_gate * activity_gate
   - cap = synapse_update_cap / (1 + depth_clip_scaling*(L/max_wander_depth))
   - w ← clip(w + update, ±weight_limit)
   - potential ← min(synapse_potential_cap, potential + |scaled|*gradient_score_scale)
   - score = |e|*|w|/max(L,1)*(1+memory_gate)
4. Accumulate updates for gradient_accumulation_steps then apply and optionally weight_decay.
5. Store paths with error below episodic_memory_threshold into episodic_memory.
6. GPU path uses kernel nb_apply_launcher implementing the same operations in parallel.
7. After each training step decay memory gates:
   gate_s <- gate_s * memory_gate_decay; remove gates when gate_s < 1e-6.

### 4.6 Structural plasticity
- For each traversed synapse with potential ≥ plasticity_threshold and dropout check passed:
  - Create new neuron with representation (source.rep + target.rep)/2 and tier promotion (vram→ram→disk).
  - w1 = syn.weight * struct_weight_multiplier1 * mod * structural_learning_rate * (1 + rep_sim).
  - w2 = syn.weight * struct_weight_multiplier2 * mod * structural_learning_rate * (1 + rep_sim).
  - Add synapse source→new_id with weight w1 and new_id→target with weight w2.
  - Remove original synapse and log event with timestamp and potential.
- Merge similar synapses if weight difference < merge_tolerance.
- Prune synapses every synapse_prune_interval steps when potential < synapse_potential_cap.
### 4.7 Attention and fatigue
- Update type-specific attention:
  a' = a * attention_decay + attention_update_scale * activity.
- Synaptic fatigue:
  f' = f * fatigue_decay + fatigue_increase * activity.
- Attention span threshold controls dynamic span module.
- Context-aware attention layer computes
  \(\text{softmax}(((Q+ctx)(K+ctx)^T)/\sqrt{d})\) with optional chaotic gating and causal masks.

### 4.8 Caching and parallel wandering
- `dynamic_wander_parallel` spawns multiple processes using `mp.Pool` with
  seeds; main process replays each path to apply updates.
- `chaotic_memory_replay` perturbs inputs via logistic map
  \(x_{n+1} = r x_n (1-x_n)\) and concatenates resulting paths.
- Wander cache stores tuples `(output, path, timestamp)` and expires entries
  after `wander_cache_ttl`, evicting LRU when exceeding `_cache_max_size`.
- Exploration bonus decays each activation:
  `exploration_bonus <- exploration_bonus * exploration_decay`.

### 4.9 Synapse-type attention and actions
- Track cumulative loss, speed and size attention per synapse type.
- `decide_synapse_action` creates a new synapse of the highest-loss type or
  removes one of the highest-size type.
- Gradient pruning:
  - `compute_gradient_prune_mask(r)` selects the lowest \(r\) fraction of
    gradients using `np.partition`.
  - `apply_gradient_prune_mask` deletes flagged synapses and associated
    optimiser state.

### 4.10 Dataset-aware training modes
- `refresh_on_dataset_change` monitors directories via `DatasetWatcher` and
  resets learning state when files change.
- `train_streaming_shards` iterates through `BitTensorDataset` shards using
  `StreamingDatasetStep` with asynchronous prefetching.
- `contrastive_train` applies InfoNCE to augmented batches; `imitation_train`
  performs behaviour cloning on demonstration pairs.

### 4.11 State management
- `to_dict`/`from_dict` and JSON helpers serialise Neuronenblitz without the
  core object.
- `reset_learning_state` clears momentum, eligibility traces, caches and replay
  buffers, ensuring fresh training runs.
- Dynamic attention span `_apply_attention_span` masks path elements according
  to `span_module` outputs.

### 4.12 Neurogenesis coupling
- Neuronenblitz exposes neuron type preferences to `MarbleBrain` for growth decisions.
- During neurogenesis, the brain queries either
  \(t^* = \text{get\_preferred\_neuron\_type}()\) or
  combined attention \(t^*_{comb}\) when using multiple spans.
- Newly created neurons and synapses are initialized with representation noise
  \(\mathcal{N}(0, \text{representation\_noise\_std})\) and weight range
  \([\text{weight\_init\_min}, \text{weight\_init\_max}]\).

### 4.13 Reinforcement learning interface
 - `enable_rl` and `disable_rl` toggle intrinsic Q-learning which encodes
   state-action pairs via `q_encoding(s,a)`.
 - `rl_select_action` implements ε-greedy policy:
   \(a = \begin{cases} \text{rand}(A) & \text{if } r<\epsilon \\ \arg\max_a Q(s,a) & \text{otherwise} \end{cases}\).
 - `rl_update` performs Q-learning target
   \(Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]\)
   via `train` on the encoded pair and decays ε.
 - `enable_sac` attaches actor/critic networks with optional entropy tuning;
   `sac_select_action` samples from the actor and `sac_update` minimises
   \(L_Q = \|Q(s,a) - (r + \gamma (\min(Q_1',Q_2') - \alpha \log π(a'|s')))\|^2\)
   and \(L_π = (\alpha \log π(a|s) - Q(s,a))\), tracking policy entropy.
 - `plot_sac_entropy` writes entropy curves to disk for diagnostics.

### 4.14 Optimization and scheduling
- RMSProp accumulator for gradient g:
  v' = beta * v + (1-beta) * g^2;
  g_adj = g / sqrt(v' + grad_epsilon).
- Learning rate scheduler:
  lr_{t+1} = clip(lr_t * scheduler_gamma, min_learning_rate, max_learning_rate).
- Epsilon scheduler analogous to learning rate scheduler.

### 4.15 Chaotic gating and phase adaptation
- Chaotic gate using logistic map:
  c_{t+1} = chaotic_gating_param * c_t * (1 - c_t).
- Phase update:
  phi_{t+1} = phi_t + phase_rate + phase_adaptation_rate * e.

### 4.16 Experience replay and memory gating
- Prioritized replay probability:
  P_i = (p_i^replay_alpha) / sum_j (p_j^replay_alpha).
- Importance weight:
  w_i = (1 / (N * P_i))^replay_beta.
- Memory gate value:
  g' = g * memory_gate_decay + memory_gate_strength * abs(e).
- Episodic path replay occurs with probability `episodic_memory_prob` for up to `episodic_sim_length` steps, applying the same
  synapse side effects as normal wandering.

### 4.17 Parameter inventory
All parameters of Neuronenblitz must be exposed and exercised:
backtrack_probability, consolidation_probability, consolidation_strength, route_potential_increase, route_potential_decay, route_visit_decay_interval, alternative_connection_prob, split_probability, merge_tolerance, combine_fn, loss_fn, loss_module, weight_update_fn, validation_fn, plasticity_threshold, continue_decay_rate, struct_weight_multiplier1, struct_weight_multiplier2, attention_decay, max_wander_depth, learning_rate, weight_decay, dropout_probability, dropout_decay_rate, exploration_decay, reward_scale, stress_scale, auto_update, dataset_path, remote_fallback, noise_injection_std, dynamic_attention_enabled, backtrack_depth_limit, synapse_update_cap, structural_plasticity_enabled, backtrack_enabled, loss_scale, exploration_bonus, synapse_potential_cap, attention_update_scale, attention_span_threshold, max_attention_span, span_module, plasticity_modulation, wander_depth_noise, reward_decay, synapse_prune_interval, gradient_prune_ratio, structural_learning_rate, remote_timeout, gradient_noise_std, min_learning_rate, max_learning_rate, top_k_paths, parallel_wanderers, parallel_update_strategy, beam_width, synaptic_fatigue_enabled, fatigue_increase, fatigue_decay, lr_adjustment_factor, lr_scheduler, scheduler_steps, scheduler_gamma, epsilon_scheduler, epsilon_scheduler_steps, epsilon_scheduler_gamma, momentum_coefficient, reinforcement_learning_enabled, rl_discount, rl_epsilon, rl_epsilon_decay, rl_min_epsilon, entropy_epsilon_enabled, shortcut_creation_threshold, use_echo_modulation, wander_cache_ttl, phase_rate, phase_adaptation_rate, chaotic_gating_enabled, chaotic_gating_param, chaotic_gate_init, context_history_size, context_embedding_decay, emergent_connection_prob, concept_association_threshold, concept_learning_rate, weight_limit, wander_cache_size, plasticity_history_size, rmsprop_beta, grad_epsilon, use_experience_replay, replay_buffer_size, replay_alpha, replay_beta, replay_batch_size, exploration_entropy_scale, exploration_entropy_shift, gradient_score_scale, memory_gate_decay, memory_gate_strength, episodic_memory_size, episodic_memory_threshold, episodic_memory_prob, curiosity_strength, depth_clip_scaling, forgetting_rate, structural_dropout_prob, gradient_path_score_scale, use_gradient_path_scoring, rms_gradient_path_scoring, activity_gate_exponent, subpath_cache_size, gradient_accumulation_steps, wander_anomaly_threshold, wander_history_size, subpath_cache_ttl, monitor_wander_factor, monitor_epsilon_factor, episodic_sim_length, use_mixed_precision, remote_client, torrent_client, torrent_map, metrics_visualizer.

## 5. Learning Modules
For each learning paradigm below, reimplement training loops, loss functions, evaluation metrics and configuration hooks.

### 5.1 Contrastive Learning
- Implement InfoNCE loss:
  L = -(1/N) * sum_{i=1}^N log( exp(sim(z_i, z_i_plus)/tau) / sum_{j=1}^{2N} exp(sim(z_i, z_j)/tau) ).
- Integrate with ContrastiveLearner and pipeline.

### 5.2 Imitation Learning
- Behavioural cloning with categorical cross-entropy:
  \(L = -\sum_i y_i \log p_i\).
- Integrate dataset loader and ImitationLearner.

### 5.3 Autoencoder Learning
- Reconstruction loss \(L_{rec} = \|x - \hat{x}\|_2^2\).
- Variational regularisation \(L_{KL} = -\tfrac{1}{2}\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)\).
- Denoising input perturbation: \(\tilde{x} = x + \mathcal{N}(0, \sigma^2)\)
  with \(\sigma_{t+1} = \sigma_t \cdot \text{noise\_decay}\).
- Training step: `dynamic_wander` on noisy input, compute error `e = x - \hat{x}`,
  apply `apply_weight_updates_and_attention`, then `perform_message_passing`.
- Config section exposes `enabled`, `epochs`, `batch_size`, `noise_std` and `noise_decay`; training is skipped when disabled and defaults are applied when not provided.
- `AutoencoderPipeline` converts arbitrary objects to floats via
  `DataLoader` and optional `Tokenizer`, collects values and persists
  `{core, neuronenblitz}` using pickle, handling numeric arrays, text with trained tokenizers, existing `BitTensorDataset` instances or auto-generated vocabularies when `use_vocab=True`.

### 5.4 Reinforcement Learning
- Q-learning update: \(Q(s,a) \leftarrow Q(s,a) + \alpha (r + \gamma \max_{a'}Q(s',a') - Q(s,a))\).
- Policy gradient objective: \(\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) (G_t - b)]\).
- Soft Actor-Critic losses:
  - Critic: \(L_Q = \|Q_\phi(s,a) - (r + \gamma(\min_i Q_{\bar{\phi}_i}(s',a') - \alpha \log \pi_\theta(a'|s')) )\|_2^2\).
  - Actor: \(L_\pi = \mathbb{E}[\alpha \log \pi_\theta(a|s) - Q_\phi(s,a)]\).
- Integrate replay buffer, exploration scheduling and reward scaling.

### 5.5 Curriculum and Transfer Learning
- Implement schedulers for task difficulty and transfer pipelines.
- Gradually adjust task distribution D_t to D_{t+1} such that KL(D_{t+1} || D_t) < epsilon.

### 5.6 Meta and Continual Learning
- Implement meta-parameter controller adjusting plasticity using validation losses.
- Continual learning:
  - Elastic weight consolidation: penalty \(\sum_i \frac{\lambda}{2} F_i (\theta_i-\theta_i^*)^2\).
  - Replay buffer based learner storing at most `memory_size` samples and
    rehearsing one random example each step.

### 5.7 Adversarial and Fractal Learners
- GAN objective: \(\min_G \max_D E_{x\sim p_{data}}[\log D(x)] + E_{z\sim p_z}[\log(1 - D(G(z)))]\).
- FGSM adversarial examples: x_adv = x + \epsilon \cdot \mathrm{sign}(\nabla_x L(model(x), y)).
- FGSMDataset wraps datasets to generate x_adv on-the-fly; training loop updates discriminator and generator via Neuronenblitz dynamic_wander.
- Adversarial training loop for PyTorch models perturbs inputs and trains on (x_adv, y).
- Fractal dimension learning uses correlation dimension \(D_2 = \lim_{r\to 0} \frac{\log C(r)}{\log r}\).
### 5.8 Harmonic Resonance and Quantum Flux
- Harmonic resonance loss using Fourier transforms to match frequency spectra.
- Quantum flux learning uses complex-valued amplitudes with unitary constraint U^dagger U = I.
- `QuantumFluxPairsPipeline` trains on numeric or textual pairs, accepts optional tokenizers via `DataLoader`, supports `BitTensorDataset` inputs and can auto-build vocabularies when `use_vocab` is true.

### 5.9 Synaptic Echo Learning
- Implement echo-based weight consolidation where echo signal e_t is a decayed trace of past activations:
  e_{t+1} = echo_decay * e_t + activation_t.
- Weight update couples current gradient g_t with echo: \Delta w = echo_strength * e_t * g_t.

### 5.10 Continuous Weight Field Learning
- Represent weight field as radial basis functions
  \(\phi_j(x) = \exp(-(x-c_j)^2/(2\sigma^2))\) with centres \(c_j\).
- Prediction \(\hat{y} = \phi(x)^T W \phi_x\) where \(\phi_x\) is the
  neuron's representation.
- Gradient step for each dimension j:
  \(w_j \leftarrow w_j + \eta\, \phi(x)\,e\,\phi_x[j] - \eta\,\lambda\, w_j\).

### 5.11 Federated and Distributed Learning
- Implement federated averaging: w_{t+1} = \sum_k (n_k / N) * w_k where n_k are client sample counts.
- Integrate distributed_training with gradient synchronization across nodes.

### 5.12 Diffusion Models
- Forward process: \(x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\).
- Reverse denoising: learn score function \(s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)\) and integrate \(x_{t-1} = x_t + \beta_t s_\theta(x_t,t) + \sqrt{\beta_t} z\).
- Integrate diffusion_core, diffusion_pairs_pipeline and scheduler into training pipeline.

### 5.13 Conceptual and Schema Induction
- Neural schema induction mines frequent relational triples using support threshold and max schema size; expand schemas until frequency < threshold.
- Conceptual integration:
  - When random draw < blend_probability and cosine similarity between
    active neurons is below threshold, create blended neuron with
    representation \(\tanh(r_i \odot r_j)\).
  - Connect new neuron bidirectionally with its parents with unit weights.
- N-dimensional topology learner embeds neurons in \(d\)-dimensional space and optimises attention threshold \(\alpha\) s.t. loss decreases by > loss_improve_threshold within stagnation_epochs.
- Unified learning combines gated learners with log mixture objective: \(L = \log\sum_i g_i e^{-L_i}\) where gating weights \(g_i\) are softmax outputs.

### 5.14 Advanced GPT model
- Custom autograd Tensor class with operations (+, *, @, tanh, reshape, transpose).
- Transformer decoder uses self-attention with mask and feed-forward layers.
- Training loop computes cross-entropy and optional KL-divergence distillation:
  L = CE(logits, target) + distill_alpha * KL(softmax(logits) || prev_logits).
- Gradient clipping: scale = max_grad_norm / (norm + 1e-6); parameters updated by p -= lr * grad.
### 5.15 Semi-supervised Learning
- Consistency regularisation uses labeled pair \((x_l, y_l)\) and unlabeled input \(u\).
  - Supervised loss: \(L_{sup} = (y_l - f(x_l))^2\).
  - Two stochastic forward passes on \(u\) produce predictions \(f(u)\) and \(f'(u)\) with consistency loss \(L_{con} = (f(u) - f'(u))^2\).
  - Total objective: \(L = L_{sup} + \lambda L_{con}\) where \(\lambda = \text{unlabeled\_weight}\).
- Update Neuronenblitz along paths for both labeled and unlabeled passes, performing message passing after each update.
- `SemiSupervisedPairsPipeline` builds BitTensorDataset instances for labeled and unlabeled data, converts objects to floats and pickles `{core, neuronenblitz}`.
## 6. Pipelines and Orchestration
### 6.1 Pipeline framework
1. Build `HighLevelPipeline` orchestration:
   1. Represent each step as a dictionary storing `func`, `module` or `callable` and `params`.
   2. Implement management operations: `add_step`, `insert_step`, `remove_step`, `move_step`, `replace_step`, `update_step_params`, `duplicate`, `get_step`, `list_steps` and `clear_steps`.
   3. Expose module functions as pipeline steps via `_ModuleWrapper` allowing dynamic attribute access.
   4. Register dataset argument names and default `BitTensorDataset` parameters; auto-convert iterables to `BitTensorDataset` in `_maybe_bit_dataset`.
   5. Validate dependencies with `_build_dependency_graph` and enforce acyclic order using `_topological_sort`.
   6. Execute steps sequentially with `_execute_steps`:
      - handle nested `macro` pipelines and per-step caching to disk via `_cache_path`/`clear_cache`.
      - support isolated subprocess execution and tier delegation through `TIER_REGISTRY`.
      - automatically train `Neuronenblitz` on dataset results and enforce per-step memory limits using `psutil`.
   7. Mirror semantics in `_execute_steps_async` leveraging `asyncio` and thread executors for synchronous and asynchronous functions.
   8. Extract MARBLE instances and decode training pairs using `_extract_marble` and `_train_neuronenblitz`.
   9. Provide public interfaces `execute`, `execute_async`, `execute_stream`, `run_step`, `run_step_async`, `execute_until`, `execute_until_async`, `execute_from`, `execute_from_async`, `execute_range`, `execute_range_async` and `summary`.
   10. Serialise pipelines with `save_json`, `to_json`, `load_json`, `from_json` and checkpoint via `save_checkpoint`/`load_checkpoint`.
   11. Enable step freezing/unfreezing (`freeze_step`/`defrost_step`),
       per-step benchmarking with `MetricsVisualizer`, neuron/synapse
       preallocation and progress broadcast through `global_workspace`.
2. Implement `pipeline_cli`, `pipeline_schema` and example workflows.
3. Incorporate `BranchContainer` to execute sub-pipelines concurrently with device-aware scheduling, GPU memory checks and optional concurrency limits.

### 6.2 Scheduler plugins
- Implement plugin interface for custom schedulers such as dream_scheduler and remote_worker_pool.
- Ensure each scheduler option in config is wired to executable code.

### 6.3 Tool and learner plugins
- Provide `WebSearchTool` plugin invoking DuckDuckGo and expose registration hook.
- Recreate tool_manager_plugin, tool_plugins and learning_plugins with dynamic discovery.

## 7. Memory and Simulation Systems
### 7.1 Episodic simulation and dream modules
- Support `exampletrain` style auto-firing and dreaming loops with synthetic dataset utilities for reproducible tests.
- Implement episodic_simulation, `DreamReplayBuffer` and `DreamScheduler` for salience-weighted replay with instant and long-term buffers and housekeeping.
- `DreamReinforcementLearner` mixes real and dreamed experiences controlled by dream_cycles, strength and interval.

### 7.2 Prompt and attention codelets
- Implement prompt_memory, attention_codelets and attention_utils:
  - Codelets return AttentionProposal(score, content).
  - Form coalition by softmaxing scores (plus optional salience and workspace gates) and selecting top-k.
  - Winning proposals broadcast via global workspace or direct broadcast_coalition.
  - GatingLayer modulates attention weights using sine, chaotic logistic map x_{n+1}=r x_n(1-x_n) or episodic memory reward.
  - DynamicSpanModule selects attention spans where cumulative softmax ≤ threshold, enforcing max_span.

### 7.3 Self-monitoring and metrics
- Integrate `UsageProfiler` logging epoch wall time, CPU/RAM/GPU utilisation to CSV.
- Provide `ExperimentTracker` abstraction with Wandb and Kùzu implementations and event-bus attachment helper.
- Integrate `SelfMonitor` plugin maintaining an `error_history` deque and publishing `mean_error` markers to `global_workspace`.
- Expose `system_metrics.profile_resource_usage` returning CPU, RAM and GPU usage for dashboards.

### 7.4 Cognitive modules
- Reconstruct global_workspace for broadcasting salient signals across subsystems.
- Implement theory_of_mind for agent modeling using probabilistic belief updates.
- Recreate neural_pathway and neural_schema_induction for structured knowledge extraction.

- PredictiveCodingPlugin minimises reconstruction error through iterative latent updates and logs to global workspace.
- QuantumFluxLearner updates synapses via sine-modulated phase shifts and trains through QuantumFluxPairsPipeline.
## 8. Utilities and Interop
### 8.1 External framework interop
- Implement PyTorch and TensorFlow interop layers (pytorch_to_marble, torch_interop, tensorflow_interop).
  - `tensorflow_interop` provides `MarbleKerasLayer` mirroring the message-passing MLP and `tf_to_core` which copies weights back to Marble and resizes neuron representations.
- Provide model import/export (convert_model, marble_to_pytorch, torch_model_io).
- Autograd integration via MarbleAutogradLayer and TransparentMarbleLayer supporting gradient accumulation and scheduler callbacks.

- `pytorch_to_marble` traces PyTorch models with converter registries (`register_converter`, `register_function_converter`, `register_method_converter`) and provides layer converters for Linear and Conv2d, raising `UnsupportedLayerError` for unhandled modules.
### 8.2 Remote and distributed execution
- Implement remote_offload, remote_worker_pool, distributed_training and torrent-based model exchange.
- Include networkx graph export, web API and database query tools.
- networkx_interop converts cores and pipelines to NetworkX graphs, provides diff utilities and pipeline expansion.
- neural_pathway performs tensor-based BFS path finding and Plotly visualisation of highlighted routes.

- Remote hardware plugins implement `RemoteTier` base with `connect`, `offload_core`, `run_step` and `close`; include gRPC tier with retry/backoff and mock tier for local execution, loaded via `load_remote_tier_plugin`.
- RemoteBrainServer exposes `/offload`, `/process` and `/ping` HTTP endpoints with optional compression and bearer authentication; RemoteBrainClient manages retries, latency statistics, bandwidth and route optimisation.
- RemoteWandererClient and RemoteWandererServer exchange `ExplorationRequest` and `ExplorationResult` messages over `MessageBus` to coordinate distributed wandering with optional latency simulation.
### 8.3 Experiment tracking and logging
- `UsageProfiler` and experiment trackers must be wired to training loops and pipeline events.
- Integrate experiment_tracker, logging_utils and usage_profiler with configurable backends.

- RunProfiler records start/end timestamps and device for each pipeline step and writes ordered JSON traces.
### 8.4 Configuration tooling
- Offer `workflow_template_generator` producing pipeline boilerplates for classification and preprocessing examples.
- Provide command-line and GUI-free tools:
  - `config_generator` parses `yaml-manual.txt` and produces commented configs
    by mapping section keys to descriptions.
  - `config_editor` validates YAML against schema and writes timestamped
    backups before saving.
  - `config_sync_service` watches a source config and propagates changes to
    destination paths using filesystem observers.
  - `list_config_keys` enumerates YAML keys and flags unused entries through ripgrep.
  - `find_unimplemented_params` scans `CONFIGURABLE_PARAMETERS.md` for parameters absent from code.
  - `validate_config_docs` cross-checks `config.yaml` with documentation and reports missing or extra keys.
- Implement `backup_utils` with `BackupScheduler` periodically copying
  configuration directories (interval \(T\)) to timestamped destinations and
  graceful start/stop handling.

### 8.5 Security and data integrity
- Implement dataset_encryption, crypto_utils and dataset_replication with integrity checks.
- Support torrent_offload and data_compressor for secure remote transfers.

### 8.6 Remote interaction modules
- Implement `wanderer_auth` HMAC tokens and `SessionManager` for remote wanderer authentication.
- Serialize exploration commands via `wanderer_messages` dataclasses (`ExplorationRequest`, `PathUpdate`, `ExplorationResult`).
- Expose `web_api.InferenceServer` with `/infer`, `/graph`, and `/shutdown` endpoints plus optional PromptMemory and bearer-token auth.
- Implement remote_wanderer, remote_offload, remote_hardware interface and mcp_server/tool_bridge.
- MCPServer exposes `/mcp/infer` and `/mcp/context` endpoints with optional token or basic auth and PromptMemory support; MCPToolBridge forwards `/mcp/tool` requests to MessageBus agents.
- Provide web_api endpoints and database_query_tool for external control.
### 8.7 Asynchronous training utilities
- Include federated averaging trainer `FederatedAveragingTrainer` aggregating synapse weights across clients.
- ProcessManager distributes tensor tasks across processes using shared memory and multiprocessing with spawn start method, falling back to threads on failure.
- Provide hyperparameter search utilities (`grid_search`, `random_search`).
- async_transform dispatches data tasks via scheduler plugins on CPU/GPU.
- Scheduler plugins include thread-based and asyncio implementations selectable via `configure_scheduler` and retrievable with `get_scheduler`.
- AsyncGradientAccumulator schedules backward passes with `asyncio.to_thread` and applies optimizer steps every `accumulation_steps`, matching synchronous SGD on CPU or GPU.
- BranchContainer executes step sequences concurrently while enforcing `max_gpu_concurrency` based on available GPU memory.
- `Brain.start_training` launches a background thread controlled by `training_active` and `wait_for_training`, while `train_async` exposes an awaitable API.
- `Pipeline._dataset_step_indices` detects dataset-producing steps to drive automatic training loops.
### 8.8 Visualization helpers
- activation_visualization.plot_activation_heatmap stacks neuron representations and saves heatmaps.
- Attention utilities provide `GatingLayer` (modes `sine` and `chaos`), `generate_causal_mask`, `benchmark_mask_overhead`, and Plotly `mask_figure`/`gate_figure` that accept tensors from CPU or GPU.
- BackupScheduler periodically mirrors source directories to timestamped backups.
- MetricsVisualizer streams metrics to TensorBoard, CSV and JSON logs while triggering scheduled backups.

### 8.9 Command-line interface
- Add `highlevel_pipeline_cli` for checkpoint/resume operations with device selection and dataset version metadata.
- Provide `pipeline_cli` that loads pipeline JSON, reads `--config` YAML and passes `cache_dir` and `default_step_memory_limit_mb` to `Pipeline.execute`, exiting with code 0.
- Build `cli.py` supporting configuration overrides, dataset loading, training,
  evaluation, pipeline execution, hyperparameter grid search via YAML files, core export, learning-rate scheduler overrides and configuration synchronization.
- Expose options for learning-rate schedulers (with `--lr-scheduler` and `--scheduler-gamma`), parallel wanderers, cross-validation, quantization, unified learning, remote retry/backoff, message-passing benchmarks and sync interval overrides (`--sync-interval-ms`).
- `--help` output must describe all commands and print "MARBLE command line interface" header.


### 8.10 Model conversion and compression
- Incorporate `generate_repo_md` script to snapshot repository contents into single Markdown for reproducibility.
- `convert_model` CLI transforms PyTorch checkpoints to MARBLE JSON or `.marble` snapshots and offers summary output, plots, CSV, tables and graph rendering.
- `convert_pytorch_model` converts pretrained PyTorch modules and verifies prediction parity on sample datasets.
- `QuantizedTensor` enables uniform n-bit quantization with bit packing/unpacking, `state_dict` serialization and device-aware `to` transfers. Linear layers reconstructed from quantized weights must match dense outputs within tolerance, bit streams have expected length, round-trips preserve values across CPU and GPU.
- model_refresh provides full_retrain, incremental_update and auto_refresh routines triggered by DatasetWatcher.
- `DataCompressor` and crypto utilities provide constant-time XOR encryption, AES-GCM tensor/byte encryption, delta encoding, quantization and sparse-aware compression.
- `DatabaseQueryTool` executes Cypher queries on Kùzu databases.
- `core_from_json` accepts legacy snapshots lacking `synapse_type` or `potential` fields to maintain backward compatibility.

### 8.11 Training utilities
- Adversarial toolkit includes `fgsm_generate` for perturbations, `FGSMDataset` wrapping datasets to emit adversarial examples, and `AdversarialLearner` training generator/discriminator Neuronenblitz pairs while recording loss history.
- `DistillationTrainer` blends teacher predictions with targets for student brains.
- `DistributedTrainer` uses PyTorch DDP to average synapse weights across processes.
- `EvolutionTrainer` explores configuration space via mutation, parallel fitness evaluation and lineage graph export.
- ReinforcementLearning module offers GridWorld env, `MarbleQLearningAgent` with epsilon decay/Double-Q and `MarblePolicyGradientAgent` integrating Neuronenblitz outputs. Agents expose `enable_rl`, `rl_select_action` and `rl_update` with learning-rate control; training loops must improve cumulative reward and support configurable hidden dimensions.
- Interface helpers: `curriculum_train` sequencing tasks with schedule strategies, `set_dreaming` toggling dream simulation, `set_autograd` attaching `MarbleAutogradLayer` (supports gradient accumulation, schedulers and N-dimensional tensors on CPU/GPU), `convert_pytorch_model` importing PyTorch weights with prediction map, `load_hf_dataset` retrieving HuggingFace samples (optionally encoding via DataLoader), and `streaming_dataset_step` yielding prefetching iterators.
- Brain utilities persist dream buffers and neuromodulatory context via `save_model`/`load_model`, migrate legacy checkpoints with `save_checkpoint`/`load_checkpoint`, log metrics (loss, VRAM usage, neuromod signals, plasticity threshold, message passing change, compression ratio) through `MetricsVisualizer`, expose `benchmark_step` returning marble/autograd losses and times, `generate_chain_of_thought` tracing synapse paths, and respect `log_interval` during training. Training exposes `progress_callback` emitting fractional completion after each epoch.
- `SelfMonitoring` plugin tracks gradient norms, memory usage and configuration drift, publishing anomalies to `MessageBus` and triggering `SystemMetrics` logging.
### 8.12 Python compatibility utilities
- `pycompat.removeprefix` and `pycompat.cached` supply Python 3.8 helpers.
### 8.13 Tensor backend and synchronization
- `_Backend` abstraction offers NumPy and JAX implementations supporting `matmul`, `sigmoid`, `relu`, `seed`, `rand` and `randn`.
- `to_numpy` and `_copy_to` convert between NumPy, JAX and torch tensors.
- `TensorSyncService` launches per-device threads computing XOR/subtraction deltas, broadcasting them and tracking `bytes_sent`, `syncs` and `stale_resyncs` with automatic resynchronisation.
### 8.14 Sparse utilities and shared vocabulary
- `sparse_utils` converts dense arrays to CSR/CSC/COO matrices, computes byte usage and benchmarks memory savings.
- `shared_vocab.build_shared_vocab` aggregates bitstreams from multiple datasets with parameters `(min_len, max_len, max_size, start_id, min_occurrence)`.
### 8.15 Hyperparameter optimisation and synthetic datasets
- `optimize.py` runs Optuna on a tiny `FakeData` dataset, trains a small network and saves best parameters to YAML.
- `synthetic_dataset` supplies `generate_sine_wave_dataset` and `generate_linear_dataset` for quick experiments.
### 8.16 CPU fallback validation scripts
- `scan_cuda_modules` locates CUDA-dependent modules.
- `catalog_cpu_fallback_tests` emits `cpu_fallback_report.md` for modules lacking CPU tests.
- `run_cpu_fallback_tests` disables CUDA and runs each pytest file to verify CPU-only execution.
- `validate_config_docs` and `find_unimplemented_params` ensure configuration keys are implemented and documented.
- `convert_to_py38` rewrites postponed annotations for Python 3.8 compatibility.
### 8.17 Template modules for custom components
- `neuron_template`, `rnn_neuron_template`, `synapse_template` and `gating_synapse_template` illustrate how to craft custom neurons and synapses.
### 8.18 MCP server and web APIs
- Implement `mcp_server` exposing Model Control Protocol endpoints for starting, stopping and inspecting pipelines via streaming JSON-RPC over WebSockets.
- Provide `mcp_tool_bridge` translating MCP tool calls into `ToolManager` actions and relaying responses through `MessageBus`.
- Build RESTful `web_api` offering dataset management, training control, metrics retrieval and health checks; integrate authentication and CORS.
- Expose `web_search_tool` to query external sources and feed results back into pipelines via `ToolManagerPlugin`.
## 9. Testing and Validation
### 9.1 Unit tests
- Write pytest suites for every module and parameter combination.
- Ensure tests cover CPU and GPU execution paths.
- Run `run_cpu_fallback_tests` with CUDA disabled to verify CPU-only behaviour for all test files.
- Generate `cpu_fallback_report.md` via `catalog_cpu_fallback_tests` to track CUDA modules missing CPU tests.

### 9.2 Integration tests
- Simulate end-to-end pipelines verifying data flow from datasets through learners and Neuronenblitz.

### 9.3 Performance and stress tests
- Recreate benchmarks to verify parity:
  - `benchmark_autograd_vs_marble` compares pure MARBLE vs autograd layer and reports `{"marble": {"loss", "time"}, "autograd": {"loss", "time"}}`.
  - `benchmark_dream_consolidation` measures impact of dream cycles, returning `{"with_dream": {"avg_error", "duration"}, "without_dream": {"avg_error", "duration"}}`.
  - `benchmark_graph_precompile` times graph caching and yields `{"no_precompile", "precompiled", "speedup"}` for each device.
  - `benchmark_parallel_wanderers` evaluates multi-process wandering speed.
  - `benchmark_remote_wanderer_latency` simulates network delays.
  - `benchmark_sac_vs_baseline` tests SAC-enabled wanderer vs random.
  - `benchmark_super_evolution` profiles SuperEvolutionController dynamics, producing runs of 20 entries each containing `{"loss", "changes"}` where at least one change list is non-empty.

- `pytorch_challenge` contrasts Marble vs SqueezeNet on the Digits dataset, measuring loss, runtime and model size.
### 9.4 Config coverage
- Add tests asserting no orphaned configuration keys.

### 9.5 Cross-validation and hyperparameter search
- Implement k-fold cross-validation wrappers (cross_validation module) with deterministic splits and automatic tensor device movement.
- Integrate hyperparameter_search to sweep configuration spaces and record metrics.

## 10. Documentation and Tutorials
- Provide `project_template/main.py` as a minimal entry point loading config, dataset and training routine.
- Recreate comprehensive example suite covering numeric regression, image classification, remote offloading, GPT training, RNN sequence modelling, reinforcement learning variants, contrastive, attention codelets, Hebbian, adversarial, autoencoder, sklearn integration, iris classification, semi-supervised, federated, curriculum, meta, transfer, continual, imitation, harmonic resonance, synaptic echo, fractal dimension, quantum flux, dream reinforcement and text-to-music pipelines.
- Provide `exampletrain` advanced training demo integrating Stable Diffusion, auto-firing, dreaming, benchmarking and synthetic dataset utilities (`exampletrain_utils`).
- Document dataset download helpers (`download_cifar10.py`, `download_imdb.py`, `download_iris.py`) and shared vocabulary construction for reproducible data preparation.
- Regenerate README, ARCHITECTURE_OVERVIEW, ML_PARADIGMS_HANDBOOK and a new multi-project TUTORIAL without GUI references.
- README must list exactly ten backcronyms headed by "Mandelbrot Adaptive Reasoning Brain-Like Engine".
- Maintain ROADMAP, TROUBLESHOOTING, HIGHLEVEL_PIPELINE_TUTORIAL and configuration manuals.

## 11. Release and Maintenance
- Provide versioning strategy, changelog generation and automated publishing.
- Establish code-style guidelines and contribution templates.

Following this sequence will rebuild MARBLE with complete feature parity and precise parameter utilization without introducing a GUI.
