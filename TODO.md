# Roadmap for Marble, Marble Core, and Neuronenblitz Improvements

This TODO list outlines 100 enhancements spanning the Marble framework, the underlying Marble Core, and the Neuronenblitz learning system. The items are grouped by broad themes but are intentionally numbered for easy tracking.

1. [x] Expand unit test coverage across all modules.
2. [x] Implement continuous integration to automatically run tests on pushes.
3. [x] Improve error handling in `marble_core` for invalid neuron parameters.
4. [x] Add type hints to all functions for better static analysis.
   - [x] Add hints to marble_core.py
   - [x] Add hints to marble_neuronenblitz.py
   - [x] Add hints to streamlit_playground.py
5. [x] Integrate GPU acceleration into all neural computations.
   - [x] Evaluate current modules for GPU compatibility.
   - [x] Implement GPU kernels using PyTorch operations and custom CUDA if needed.
   - [x] Provide CPU fallback mechanisms.
   - [x] Add tests verifying GPU and CPU parity.
6. [x] Provide a command line interface for common training tasks.
7. [x] Refactor `marble_neuronenblitz.py` into logical submodules.
   - [x] Identify separate functionalities (learning algorithms, memory, etc.).
   - [x] Split file into modules under new package `marble_neuronenblitz/`.
   - [x] Update imports across the project.
   - [x] Add tests ensuring modules operate as before.
8. [x] Document all public APIs with docstrings and examples.
   - [x] Audit modules to list all public functions and classes.
   - [x] Add comprehensive docstrings using Google style.
   - [x] Provide minimal working examples in docs directory.
   - [x] Generate API reference via Sphinx.
9. [x] Create tutorials that walk through real-world datasets.
10. [x] Add automatic benchmarking for message-passing operations.
11. [x] Support asynchronous training loops for large-scale experiments.
12. [x] Add configuration schemas to validate YAML files.
13. [x] Expand the metrics dashboard with interactive visualizations.
14. [x] Implement a plugin system for custom neuron and synapse types.
15. [x] Create a dataset loader utility supporting local and remote sources.
16. [x] Provide PyTorch interoperability layers for easier adoption.
17. [x] Improve the `MetricsVisualizer` to log to TensorBoard and CSV.
18. [x] Add memory usage tracking to the core.
19. [x] Support dynamic resizing of neuron representations at runtime.
20. [x] Implement gradient clipping utilities within Neuronenblitz.
21. [x] Add a learning rate scheduler with cosine and exponential options.
22. [x] Document all YAML parameters in `yaml-manual.txt` with examples.
23. [x] Provide GPU/CPU fallbacks for all heavy computations.
24. [x] Add tests ensuring compatibility with PyTorch 2.7 and higher.
25. [x] Improve logging with structured JSON output.
26. [x] Implement distributed training across multiple GPUs.
   - [x] Research distributed training approaches (DDP, Horovod).
   - [x] Add distributed setup utilities to `marble_core`.
   - [x] Implement distributed training pipeline in `marble_neuronenblitz`.
   - [x] Write tests using CPU-based simulator.
27. [x] Provide higher-level wrappers for common reinforcement learning tasks.
   - [x] Design RL environment interface.
   - [x] Implement wrappers for policy gradient, Q-learning, etc.
   - [x] Add examples demonstrating wrappers.
   - [x] Document usage in README/TUTORIAL.
28. [x] Add recurrent neural network neuron types to Marble Core.
29. [x] Introduce dropout and batch normalization synapse types.
30. [x] Create a graphical configuration editor in the Streamlit GUI.
31. [x] Enhance the GUI with dark/light mode and mobile layout tweaks.
32. [x] Add a data pre-processing pipeline with caching.
33. [x] Integrate a remote experiment tracker (e.g., Weights & Biases).
34. [x] Provide example projects for image and text domains.
35. [x] Implement a caching layer for expensive computations.
36. [x] Expand YAML configuration to allow hierarchical experiment setups.
37. [x] Add early stopping based on validation metrics.
38. [x] Provide utilities for synthetic dataset generation.
39. [x] Implement curriculum learning helpers in Neuronenblitz.
40. [x] Document best practices for hyperparameter tuning.
41. [x] Improve remote offload logic with retry and timeout strategies.
42. [x] Add robust serialization for checkpointing training state.
43. [x] Integrate a progress bar for long-running operations.
44. [x] Expand the `examples` directory with end‑to‑end scripts.
45. [x] Provide conversion tools between Marble Core and other frameworks.
46. [x] Implement an extensible metrics aggregation system.
47. [x] Improve code style consistency with automated formatting checks.
48. [x] Add support for quantization and model compression.
49. [x] Implement a plugin-based remote tier for custom hardware.
   - [x] Define plugin API for remote hardware tiers.
   - [x] Implement sample plugin using gRPC to remote server.
   - [x] Update configuration to select remote tiers.
   - [x] Add tests for plugin initialization and data transfer.
50. [x] Create visualization utilities for neuron activation patterns.
51. [x] Add parameter scheduling for exploration/exploitation trade-offs.
52. [x] Support hierarchical reinforcement learning in Neuronenblitz.
   - [x] Research HRL algorithms applicable to the architecture.
   - [x] Implement high-level action controller.
   - [x] Add low-level policy modules.
   - [x] Provide example training script.
53. [x] Implement efficient memory management for huge graphs.
   - [x] Identify memory-heavy structures.
   - [x] Implement streaming / chunking of graph data.
   - [x] Add memory pooling and reference counting.
   - [x] Benchmark and optimize.
54. [x] Add checks for NaN/Inf propagation throughout the core.
55. [x] Provide an option to profile CPU and GPU usage during training.
56. [x] Integrate dataset sharding for distributed training.
57. [x] Create a cross-platform installer script.
58. [x] Provide a simple web API for remote inference.
59. [x] Add command line tools to export trained models.
60. [x] Implement automatic synchronization of config files across nodes.
   - [x] Design synchronization protocol (e.g., using rsync or file watchers).
   - [x] Implement config sync service.
   - [x] Add CLI command to trigger manual sync.
   - [x] Write tests simulating multi-node environment.
61. [x] Enhance constant-time operations for cryptographic safety.
   - [x] Profile existing cryptographic operations.
   - [x] Replace variable-time functions with constant-time equivalents.
   - [x] Add unit tests verifying timing does not leak secrets.
   - [x] Document cryptographic safety guidelines.
62. [x] Add more comprehensive adversarial training examples.
   - [x] Implement adversarial example generators.
   - [x] Add training loops demonstrating adversarial robustness.
   - [x] Provide dataset wrappers for adversarial data.
   - [x] Document new examples in TUTORIAL.
63. [x] Provide utilities for automatic dataset downloading and caching.
64. [x] Integrate a simple hyperparameter search framework.
65. [x] Add tests verifying deterministic behaviour with fixed seeds.
66. [x] Improve readability of configuration files with comments and sections.
   - [x] Group related config parameters into sections.
   - [x] Add descriptive comments for each parameter.
   - [x] Provide script to auto-generate sample config with comments.
   - [x] Update YAML manual accordingly.
67. [x] Implement graph pruning utilities to remove unused neurons.
68. [x] Create a repository of reusable neuron/synapse templates.
69. [x] Add support for mixed precision training when GPUs are available.
70. [x] Provide dynamic graph visualisation within the GUI.
71. [x] Implement scheduled backups of experiment logs and results.
72. [x] Add a compatibility layer for older Python versions where feasible.
   - [x] Identify features incompatible with Python 3.8 and 3.9.
   - [x] Implement polyfills or wrappers.
   - [x] Setup CI matrix to test older versions.
   - [x] Document limitations.
73. [x] Provide self-contained Docker images for reproducibility.
74. [x] Implement offline mode with pre-packaged datasets.
75. [x] Add automated packaging to publish releases on PyPI.
   - [x] Create setup.py and pyproject.toml for packaging.
   - [x] Setup CI workflow for building and uploading to TestPyPI.
   - [x] Add versioning scheme.
   - [x] Document release process.
76. [x] Improve data compression for network transfers.
77. [x] Incorporate gradient accumulation for large batch training.
   - [x] Modify training loop to accumulate gradients across steps.
   - [x] Expose accumulation steps via config.
   - [x] Update scheduler and optimizer logic.
   - [x] Add tests verifying behavior.
78. [x] Add performance regression tests for critical functions.
79. [x] Integrate basic anomaly detection on training metrics.
80. [x] Expand the scheduler with cyclic learning rate support.
81. [x] Implement custom weight initialisation strategies.
82. [x] Provide a structured logging interface for the GUI.
83. [x] Add latency tracking when using remote tiers.
84. [x] Implement automatic graph visualisation for debugging.
85. [x] Provide wrappers to convert Marble models to ONNX.
86. [x] Improve the hybrid memory system for balanced usage.
87. [x] Add a mechanism to export and import neuron state snapshots.
88. [x] Document the mathematics behind synaptic echo learning.
89. [x] Implement context-aware attention mechanisms.
   - [x] Research existing attention mechanisms.
   - [x] Design architecture for context-aware attention.
   - [x] Implement module in Neuronenblitz.
   - [x] Provide unit tests and example usage.
90. [x] Add unit tests ensuring backward compatibility between versions.
91. [x] Create a `core_benchmark.py` script for micro benchmarks.
92. [x] Provide a template repository for new Marble-based projects.
93. [x] Add interactive tutorials in Jupyter notebooks.
94. [x] Expand the remote offload module with bandwidth estimation.
95. [x] Implement dynamic route optimisation in Neuronenblitz.
96. [x] Add anomaly detection for wandering behaviour.
   - [x] Define metrics to measure wandering.
   - [x] Implement anomaly detection algorithm.
   - [x] Integrate with training logs.
   - [x] Add tests for detection accuracy.
97. [x] Provide visual feedback for training progress in Streamlit.
98. [x] [x] Offer integration examples with existing ML libraries.
99. [x] Enhance documentation with troubleshooting guides.
100. [x] Establish a long-term roadmap with release milestones.

101. [x] Implement a **Global Workspace** plugin to broadcast conscious contents across all modules.
   - [x] Define data structures for global workspace broadcast.
   - [x] Implement plugin with message queue.
   - [x] Expose configuration options.
   - [x] Add tests verifying broadcast across modules.
102. [x] Add **attention codelet** plugins that form coalitions and submit them to the Global Workspace.
   - [x] Create plugin interface for attention codelets.
   - [x] Implement coalition formation logic.
   - [x] Connect with Global Workspace plugin.
   - [x] Add example configuration.
103. [x] Create a **Self-Monitoring** plugin that maintains an internal state model and meta-cognitive evaluations.
   - [x] Design internal state data structures.
   - [x] Implement self-monitoring algorithms.
   - [x] Connect output to context_history.
   - [x] Write tests for self-monitoring updates.
104. [x] Integrate higher-order thought markers from Self-Monitoring into `context_history`.
   - [x] Extend context_history data structure.
   - [x] Add functions to log HOT markers.
   - [x] Update Self-Monitoring plugin to emit markers.
   - [x] Add tests verifying markers saved.
105. [x] Link Self-Monitoring feedback to reinforcement learning and `dynamic_wander` adjustments.
   - [x] Create interface between self-monitoring and RL modules.
   - [x] Adjust dynamic_wander parameters based on self-monitoring output.
   - [x] Provide configuration hooks.
   - [x] Add integration tests.
106. [x] Implement an **Episodic Memory** plugin supporting transient buffers, long‑term storage and context-based retrieval.
   - [x] Design schemas for episodic entries.
   - [x] Implement transient buffer and long-term storage.
   - [x] Add retrieval API with context queries.
   - [x] Write tests for storing and retrieving episodes.
107. [x] Provide an **episodic simulation** plugin to replay stored episodes for planning.
   - [x] Implement episode selection and playback engine.
   - [x] Integrate with planning modules.
   - [x] Provide configuration options for simulation length.
   - [x] Document usage.
108. [x] Develop a **Goal Manager** plugin handling hierarchical goals and conflict resolution with active inference.
   - [x] Define goal hierarchy structures.
   - [x] Implement conflict resolution algorithms.
   - [x] Connect with reinforcement learning modules.
   - [x] Add tests for typical goal scenarios.
109. [x] Build a **Theory of Mind** plugin using character, mental-state and prediction subnets.
   - [x] Research ToM models suitable for integration.
   - [x] Implement subnets for character modelling and prediction.
   - [x] Connect ToM to Global Workspace and Self-Monitoring.
   - [x] Add example training script.
110. [x] Implement a **Predictive Coding** plugin offering hierarchical predictions and active inference loops.
   - [x] Design predictive coding architecture.
   - [x] Implement hierarchical prediction modules.
   - [x] Integrate with reinforcement learning and episodic memory.
   - [x] Provide tests verifying prediction accuracy.
111. [x] Expand `context_history` and `replay_buffer` to store internal markers, goals and ToM information.
   - [x] Extend data structures to include markers, goals, ToM.
   - [x] Update save/load logic.
   - [x] Add migration for old checkpoints.
   - [x] Write tests for new buffer behavior.
112. [x] Extend attention mechanisms to interface with the Global Workspace and plugin salience scores.
   - [x] Modify attention modules to accept salience inputs.
   - [x] Connect to Global Workspace for broadcast.
   - [x] Provide weight tuning parameters.
   - [x] Add tests for attention with salience.
113. [x] Add YAML configuration options for all new plugins and document them thoroughly.
114. [x] Create unit and integration tests ensuring each plugin works on CPU and GPU.
115. [x] Update tutorials and manuals with instructions on using the consciousness plugins.
   - [x] Write step-by-step tutorial for each new plugin.
   - [x] Update yaml-manual with plugin parameters.
   - [x] Add troubleshooting section.
   - [x] Provide example configuration files.
116. [x] Complete remaining items from ``dataupgradetodo.md``
   - [x] Extend dataset loading utilities to leverage the enhanced ``DataLoader``.
   - [x] Add migration tests for old checkpoints without embedded tokenizers.
   - [x] Document tokenizer training workflow in more detail.

117. [x] Add asynchronous streaming loader in `BitTensorDataset` integrating with the remote offload service.
118. [x] Introduce pluggable compression modules so the pipeline can swap algorithms transparently.
119. [x] Implement encryption of stored objects using Marble Core cryptographic utilities.
120. [x] Provide deduplication to avoid storing identical bit streams across datasets.
121. [x] Add index generation for constant-time retrieval integrated with the memory pool.
122. [x] Develop shared vocabulary management so multiple datasets keep a unified encoding.
123. [x] Integrate GPU-accelerated encoding and decoding using core operations.
124. [x] Enable background prefetching and caching to support asynchronous pipelines.
    - [x] Add `prefetch_dataset` helper running downloads in a background thread.
    - [x] Integrate the prefetch queue with pipeline execution so steps can await availability.
125. [x] Implement dataset merging with conflict resolution logic.
126. [x] Support deterministic splitting into train, validation and test sets via hashing.
127. [x] Offer dataset versioning with reversible diffs to update existing sets.
128. [x] Provide an interactive dataset browser in the Streamlit GUI for manual review.
129. [x] Stream data directly from compressed archives without extraction.
130. [x] Add a bit-level augmentation pipeline for flipping and noisy bits.
131. [x] Verify data integrity with checksums relying on marble core utilities.
132. [x] Automatically prune invalid or corrupted entries with callback hooks.
133. [x] Cache encoded bitstreams on disk for fast reload between runs.
134. [x] Coordinate dataset memory usage with Marble Core's `MemoryPool`.
    - 134a. [x] Allocate dataset pair objects from a shared MemoryPool.
    - 134b. [x] Implement helper to release datasets back into the pool.
135. [x] Execute transformations asynchronously during idle GPU cycles.
136. [x] Shard datasets for distributed training using core distributed helpers.
137. [x] Allow in-place patching of datasets while training is running.
138. [x] Manage encryption keys through pipeline configuration files.
139. [x] Adapt vocabulary dynamically when new words appear during training.
140. [x] Audit data integrity through checksums and object hashes.
141. [x] Provide a plugin system for custom object encoders and decoders.
142. [x] Support memory-mapped files so huge datasets fit into RAM.
143. [x] Track modification history with the ability to revert changes.
144. [x] Offer undo and redo commands for interactive dataset editing.
145. [x] Fetch missing remote files automatically when constructing datasets.
146. [x] Enable sample-level transformations such as image rotations or text cleanup.
147. [x] Lazily decode objects so they are materialised only when accessed.
148. [x] Select compression algorithms through a pluggable interface.
149. [x] Write datasets asynchronously so saves never block the main loop.
150. [x] Track dependencies between dataset entries and their original sources.
151. [x] Map every object to a hashed identifier for cross-dataset linking.
152. [x] Cache downloads on the network layer to avoid repeated transfers.
153. [x] Visualise bit patterns interactively via the Streamlit GUI.
154. [x] Filter samples with user-defined expressions evaluated on load.
155. [x] Detect corrupted compression files and attempt automatic recovery.
156. [x] Export and import datasets from standard formats like JSON or CSV.
157. [x] Replicate datasets across nodes with progress notifications.
158. [x] Summarise datasets in pipeline descriptions for easier debugging.
159. [x] Notify the memory manager about upcoming dataset allocations.
160. [x] Add an API to append data incrementally with vocabulary updates.
161. [x] Register debugging hooks for inspecting individual samples in the pipeline.
162. [x] Provide approximate nearest neighbour search over bit tensors for retrieval.
163. [x] Attach hierarchical tags or labels alongside each stored pair.
164. [x] Ensure cross-platform serialisation for dataset portability.
165. [x] Allow custom serialisation formats beyond pickle.
166. [x] Emit lifecycle events so the metrics visualiser can show dataset operations.
167. [ ] Enable asynchronous step execution in `HighLevelPipeline` to overlap data loading and training.
    - [x] Add `_execute_steps_async` helper handling coroutine functions and thread offloading.
    - [x] Implement `execute_async` public method executing the entire pipeline asynchronously.
    - [x] Unit tests verifying asynchronous execution with mixed coroutine and blocking steps.
    - [x] Expose `pipeline.async_enabled` configuration option with CPU and GPU support.
    - [ ] Benchmark asynchronous execution on CPU and GPU to quantify speedup.
        - [ ] Establish synchronous baseline timings on CPU.
        - [ ] Measure asynchronous execution timings on CPU.
        - [ ] Establish synchronous baseline timings on GPU.
        - [ ] Measure asynchronous execution timings on GPU.
        - [ ] Compile and analyse speedup results.
    - [x] Document asynchronous pipeline usage in README and TUTORIAL.
    - [ ] Add integration tests for asynchronous execution in multi-node environments.
        - [ ] Configure multi-node CPU environment for testing.
        - [ ] Configure multi-node GPU environment for testing.
        - [ ] Verify asynchronous steps execute correctly across nodes.
168. [ ] Cache intermediate results so iterative experiments run faster.
    - [x] Create file based cache storing step outputs keyed by index and function name.
    - [x] Add `clear_cache` method to remove cached files when needed.
    - [x] Unit tests demonstrating that repeated runs reuse cached results.
    - [x] Make cache directory configurable via `pipeline.cache_dir` with CPU/GPU aware paths.
    - [x] Track and display cache hit/miss statistics in the metrics dashboard.
    - [x] Document caching workflow and disk space considerations.
    - [ ] Stress-test cache performance on large datasets for CPU and GPU runs.
        - [ ] Prepare representative large dataset for benchmarking.
        - [ ] Run cache stress tests on CPU and record metrics.
        - [ ] Run cache stress tests on GPU and record metrics.
        - [ ] Compare results and document performance findings.
169. [x] Support checkpointing and resuming pipelines with dataset version tracking.
    - [x] Track `dataset_version` within `HighLevelPipeline` instances.
    - [x] Implement `save_checkpoint` and `load_checkpoint` methods.
    - [x] Test saving and loading pipelines with version metadata.
    - [x] Provide CLI commands to create and resume checkpoints.
        - [x] Add command to generate checkpoints from running pipelines.
        - [x] Add command to resume pipelines from saved checkpoints.
        - [x] Include CPU/GPU flags and usage help.
    - [x] Validate checkpoints across CPU and GPU environments.
        - [x] Test saving and loading on CPU-only setups.
        - [x] Test saving and loading on GPU-enabled setups.
        - [x] Ensure outputs match across devices.
    - [x] Document versioned checkpoint workflow in README and TUTORIAL.
        - [x] Explain checkpoint commands in README.
        - [x] Add tutorial section demonstrating save and resume.
        - [x] Review documentation for clarity.
    - [x] Add integration tests simulating interrupted runs and resume behaviour.
        - [x] Interrupt and resume training on CPU.
        - [x] Interrupt and resume training on GPU.
        - [x] Confirm dataset versions persist across resumes.
170. [ ] Provide interactive step visualisation in the Streamlit GUI using dataset introspection.
    - [x] Add a "Step Visualisation" expander showing step parameters and dataset info.
    - [x] Unit tests ensuring the new expander appears in the Pipeline tab.
    - [x] Render real-time metrics within each step visualisation for CPU and GPU runs.
        - [x] Stream metric updates during CPU executions.
        - [x] Stream metric updates during GPU executions.
        - [x] Refresh visualisation panels live.
    - [x] Allow exporting step details as JSON or CSV from the GUI.
        - [x] Add export buttons for JSON and CSV formats.
        - [x] Implement JSON serialisation of step details.
        - [x] Implement CSV serialisation of step details.
        - [x] Verify downloads work on desktop and mobile.
    - [x] Document visualisation features in README and TUTORIAL.
        - [x] Update README with screenshots and explanations.
        - [x] Extend tutorial with step-by-step visualisation guide.
        - [x] Mention export options and metrics display.
    - [x] Add GUI tests verifying export functionality and mobile layout.
        - [x] Test export features on desktop layout.
        - [x] Test export features on mobile layout.
        - [x] Assert metric panels render correctly in both views.
171. [x] Offer a plugin system so users can register custom pipeline steps easily.
   - [x] Draft plugin interface and lifecycle expectations.
       - [x] Outline core lifecycle phases (initialise, execute, teardown).
       - [x] Specify required methods and expected inputs/outputs.
   - [x] Implement registry and dynamic loader for third-party plugins.
       - [x] Create registry mapping identifiers to plugin classes.
       - [x] Implement loader scanning entry points or directories.
   - [x] Support CPU and GPU execution contexts within plugins.
       - [x] Detect available device and select execution target.
       - [x] Route tensors and operations to the appropriate device.
   - [x] Ship an example plugin demonstrating registration and usage.
       - [x] Build minimal plugin exercising the interface.
       - [x] Provide instructions to register and invoke the sample.
   - [x] Add unit and integration tests for plugin discovery and execution.
       - [x] Test registry registration and lookup behaviour.
       - [x] Execute example plugin on CPU and GPU to verify paths.
   - [x] Document plugin architecture and example workflow in README and TUTORIAL.
       - [x] Add architecture overview section to README.
       - [x] Extend tutorial with step-by-step plugin creation guide.
172. [x] Manage dependencies between steps automatically to maintain correct order.
   - [x] Design dependency graph representation for pipeline steps.
       - [x] Choose data structure to model nodes and edges.
       - [x] Document semantics of dependencies and allowed patterns.
   - [x] Implement topological sorting with cycle detection and clear errors.
       - [x] Implement sorting algorithm with cycle checks.
       - [x] Raise descriptive error messages when cycles are found.
   - [x] Ensure resolved ordering executes correctly on CPU and GPU.
       - [x] Run sample pipelines on CPU and validate outputs.
       - [x] Run sample pipelines on GPU to confirm parity.
   - [x] Add unit and integration tests for dependency management.
       - [x] Test graph construction and sorting logic in isolation.
       - [x] Create integration tests covering complex dependency chains.
   - [x] Document dependency configuration and troubleshooting in README and TUTORIAL.
       - [x] Provide examples of declaring step dependencies.
       - [x] Include troubleshooting tips for cycle and ordering issues.
173. [x] Allow branching paths in a pipeline to explore alternative experiment flows.
   - [x] Design branch step abstraction and merge semantics.
       - [x] Define API for creating branch and merge nodes.
       - [x] Specify rules for combining outputs from branches.
   - [x] Implement branching container enabling parallel sub-pipelines.
       - [x] Develop container managing branch execution contexts.
       - [x] Ensure synchronization points after branch completion.
   - [x] Handle CPU/GPU resource allocation across concurrent branches.
       - [x] Plan device assignment for branch workloads.
       - [x] Monitor memory usage to avoid overcommit.
   - [x] Add tests verifying branch execution, merging, and error handling.
       - [x] Cover normal branch merging scenarios.
       - [x] Simulate failures within branches to test recovery.
   - [x] Document branching usage with illustrative examples in README and TUTORIAL.
       - [x] Provide code sample demonstrating two-branch pipeline.
       - [x] Explain when branching is advantageous.
174. [x] Send real-time progress events to the GUI during pipeline execution.
   - [x] Define event schema and integrate with existing message bus.
       - [x] Specify fields included in progress events.
       - [x] Wire schema into current message bus definitions.
   - [x] Implement progress emitter in pipeline core with CPU/GPU hooks.
       - [x] Emit events at key step boundaries.
       - [x] Include device information in emitted data.
   - [x] Update Streamlit GUI to subscribe and render live progress updates.
       - [x] Register listener consuming progress events.
       - [x] Display updates in desktop and mobile layouts.
   - [x] Add tests checking event emission and GUI rendering on desktop and mobile.
       - [x] Unit test event publishing mechanisms.
       - [x] GUI tests verifying updates appear in both form factors.
   - [x] Document progress event workflow in README and TUTORIAL.
       - [x] Describe event flow from pipeline to GUI.
       - [x] Add troubleshooting section for missing updates.
175. [x] Recover gracefully from remote failures with retry logic.
   - [x] Specify retry policies and backoff strategies for remote calls.
       - [x] Define default retry counts and delay schedules.
       - [x] Allow customization per step or endpoint.
   - [x] Implement failure detection and retry handler compatible with CPU/GPU steps.
       - [x] Detect transient errors and trigger retries.
       - [x] Ensure retries release GPU resources properly.
   - [x] Expose configurable retry parameters via YAML and CLI.
       - [x] Add YAML fields for retry count and backoff.
       - [x] Support equivalent CLI flags.
   - [x] Add tests simulating transient and persistent remote failures.
       - [x] Simulate recoverable failures and confirm retries.
       - [x] Verify persistent failures surface clear errors.
   - [x] Document recovery scenarios and configuration in README and TUTORIAL.
       - [x] Provide examples of tuning retry parameters.
       - [x] Explain differences between transient and fatal failures.
176. [x] Execute steps on multiple processes while sharing datasets through the core.
   - [x] Design multiprocessing architecture and shared dataset mechanism.
       - [x] Decide between multiprocessing or thread pools.
       - [x] Plan shared memory or IPC for dataset access.
   - [x] Implement process manager coordinating step execution across workers.
       - [x] Spawn worker processes with configured tasks.
       - [x] Aggregate results and handle worker lifecycle.
   - [x] Ensure dataset sharing works seamlessly for CPU and GPU tensors.
       - [x] Share CPU tensors via memory maps.
       - [x] Transfer GPU tensors efficiently between processes.
   - [x] Add tests verifying multiprocessing execution and data consistency.
       - [x] Run pipeline across multiple workers on CPU.
       - [x] Repeat tests on GPU-enabled machines.
   - [x] Document multi-process setup and troubleshooting in README and TUTORIAL.
       - [x] Provide setup instructions and environment variables.
       - [x] Include guidance for debugging deadlocks or hangs.
177. [x] Add pre and post hooks for each step enabling custom behaviour.
   - [x] Define hook interfaces for actions before and after steps.
       - [x] Specify function signatures for pre and post hooks.
       - [x] Describe available context passed to hooks.
   - [x] Implement registration and ordered invocation of hooks in pipeline core.
       - [x] Provide API to register multiple hooks per step.
       - [x] Ensure hooks execute in deterministic order.
   - [x] Guarantee hooks operate correctly on CPU and GPU paths.
       - [x] Test hooks manipulating CPU tensors.
       - [x] Test hooks handling GPU tensors without leaks.
   - [x] Add tests covering hook execution and interaction with steps.
       - [x] Unit test hook registration and removal.
       - [x] Integration test hooks altering step behaviour.
   - [x] Document hook patterns with code samples in README and TUTORIAL.
       - [x] Include example demonstrating pre-processing hook.
       - [x] Highlight best practices for side-effect management.
178. [x] Provide templates to quickly generate common workflows.
   - [x] Catalogue common workflow patterns for template generation.
       - [x] Gather representative pipelines from existing projects.
       - [x] Identify configurable parameters for each pattern.
   - [x] Implement template generator producing starter pipeline code.
       - [x] Create code scaffolding with placeholders.
       - [x] Provide command-line interface to select templates.
   - [x] Include CPU and GPU configuration options within templates.
       - [x] Detect available hardware and set defaults accordingly.
       - [x] Offer flags to override device selection.
   - [x] Add tests ensuring generated templates run end-to-end.
       - [x] Generate templates and execute on CPU.
       - [x] Repeat execution on GPU if available.
   - [x] Document available templates and usage instructions in README and TUTORIAL.
       - [x] Write step-by-step guide for using generator.
       - [x] Include troubleshooting for common template issues.
179. [x] Automatically build training loops for Neuronenblitz when dataset steps are present.
   - [x] Outline design for Automatically build training loops for Neuronenblitz when dataset steps are present.
       - [x] Identify dataset step types that trigger training loop creation.
       - [x] Specify how loops detect CPU vs GPU execution paths.
       - [x] Draft flowchart illustrating automatic loop insertion points.
   - [x] Implement Automatically build training loops for Neuronenblitz when dataset steps are present with CPU/GPU support.
       - [x] Detect dataset steps during pipeline compilation.
       - [x] Instantiate training loop objects bound to detected datasets.
       - [x] Ensure created loops respect available hardware and switch between CPU and GPU.
   - [x] Add tests validating Automatically build training loops for Neuronenblitz when dataset steps are present.
       - [x] Unit test dataset step detection logic.
       - [x] Integration test auto-generated loop running on sample CPU dataset.
       - [x] Integration test loop operating on GPU when available.
   - [x] Document Automatically build training loops for Neuronenblitz when dataset steps are present in README and TUTORIAL.
       - [x] Describe automatic loop creation mechanism in README.
       - [x] Provide tutorial example demonstrating dataset-driven loop generation.
180. [x] Offer a specialised step that consumes the new streaming `BitTensorDataset`.
   - [x] Outline design for Offer a specialised step that consumes the new streaming `BitTensorDataset`.
       - [x] Define step interface for streaming dataset consumption.
         - Introduce an `StreamingDatasetStep` abstraction implementing `next_batch()` and
           `is_finished()` methods.  `next_batch()` yields a dictionary of tensors while
           `is_finished()` allows the pipeline to detect natural termination.  The step
           accepts a `BitTensorDataset` instance and exposes configuration fields for
           batch size and prefetch depth so it can be tuned through YAML and CLI.
       - [x] Plan buffering and backpressure handling for continuous data.
         - Employ an asyncio powered producer/consumer ring buffer.  The producer pulls
           from the dataset iterator and fills a bounded `asyncio.Queue`.  When the
           queue reaches capacity the dataset iterator is paused, providing natural
           backpressure.  Consumers await batches from the queue, ensuring smooth flow
           without overwhelming memory.  Overflow policies drop the oldest batch and log
           a warning to maintain forward progress.
       - [x] Detail CPU and GPU tensor flow requirements.
         - Batches emitted by the step carry device metadata.  When running on GPUs, the
           producer pins memory and performs asynchronous `to(device)` transfers before
           enqueueing.  On CPU-only systems the tensors are yielded directly.  This
           uniform interface lets downstream steps simply call `.to(current_device)` to
           guarantee correct placement while minimising unnecessary copies.
  - [x] Implement Offer a specialised step that consumes the new streaming `BitTensorDataset` with CPU/GPU support.
      - [x] Create step class wrapping BitTensorDataset iterator.
      - [x] Implement asynchronous data fetching compatible with CPU and GPU tensors.
      - [x] Integrate step into pipeline execution engine.
          - [x] Expose factory function in `marble_interface`.
          - [x] Auto-consume streams in `Pipeline.execute`.
  - [x] Add tests validating Offer a specialised step that consumes the new streaming `BitTensorDataset`.
      - [x] Unit test streaming step with synthetic dataset.
      - [x] Stress test behaviour under variable stream rates.
      - [x] Verify GPU execution matches CPU results.
   - [x] Document Offer a specialised step that consumes the new streaming `BitTensorDataset` in README and TUTORIAL.
       - [x] Explain configuration and usage in README.
       - [x] Add tutorial section showing streaming dataset pipeline.
181. [x] Persist step results to disk for quick re-runs.
   - [x] Outline design for Persist step results to disk for quick re-runs.
   - [x] Implement Persist step results to disk for quick re-runs with CPU/GPU support.
   - [x] Add tests validating Persist step results to disk for quick re-runs.
   - [x] Document Persist step results to disk for quick re-runs in README and TUTORIAL.
182. [ ] Visualise pipelines as graphs using the marble graph builder.
   - [ ] Outline design for Visualise pipelines as graphs using the marble graph builder.
   - [ ] Implement Visualise pipelines as graphs using the marble graph builder with CPU/GPU support.
   - [ ] Add tests validating Visualise pipelines as graphs using the marble graph builder.
   - [ ] Document Visualise pipelines as graphs using the marble graph builder in README and TUTORIAL.
183. [ ] Limit GPU memory usage per step through concurrency controls.
   - [ ] Outline design for Limit GPU memory usage per step through concurrency controls.
   - [ ] Implement Limit GPU memory usage per step through concurrency controls with CPU/GPU support.
   - [ ] Add tests validating Limit GPU memory usage per step through concurrency controls.
   - [ ] Document Limit GPU memory usage per step through concurrency controls in README and TUTORIAL.
184. [x] Debug steps interactively by inspecting their inputs and outputs.
   - [x] Outline design for Debug steps interactively by inspecting their inputs and outputs.
       - Introduce an ``InteractiveDebugger`` using pre and post hooks to capture
         step parameters and result summaries.
       - Summaries report tensor shape, dtype and CPU/GPU device without moving
         data between devices.
       - Expose ``Pipeline.enable_interactive_debugging`` helper registering the
         hooks and optionally dropping into ``pdb`` before and after each step.
   - [x] Implement Debug steps interactively by inspecting their inputs and outputs with CPU/GPU support.
   - [x] Add tests validating Debug steps interactively by inspecting their inputs and outputs.
   - [x] Document Debug steps interactively by inspecting their inputs and outputs in README and TUTORIAL.
185. [x] Export trained models automatically as a final pipeline step.
   - [x] Outline design for Export trained models automatically as a final pipeline step.
   - [x] Implement Export trained models automatically as a final pipeline step with CPU/GPU support.
   - [x] Add tests validating Export trained models automatically as a final pipeline step.
   - [x] Document Export trained models automatically as a final pipeline step in README and TUTORIAL.
186. [x] Log pipeline events to the remote experiment tracker.
   - [x] Outline design for Log pipeline events to the remote experiment tracker.
   - [x] Implement Log pipeline events to the remote experiment tracker with CPU/GPU support.
   - [x] Add tests validating Log pipeline events to the remote experiment tracker.
   - [x] Document Log pipeline events to the remote experiment tracker in README and TUTORIAL.
187. [x] Validate configuration of each step using marble core schemas.
   - [x] Outline design for Validate configuration of each step using marble core schemas.
   - [x] Implement Validate configuration of each step using marble core schemas with CPU/GPU support.
   - [x] Add tests validating Validate configuration of each step using marble core schemas.
   - [x] Document Validate configuration of each step using marble core schemas in README and TUTORIAL.
188. [x] Group multiple operations into macro steps for convenience.
   - [x] Outline design for Group multiple operations into macro steps for convenience.
   - [x] Implement Group multiple operations into macro steps for convenience with CPU/GPU support.
   - [x] Add tests validating Group multiple operations into macro steps for convenience.
   - [x] Document Group multiple operations into macro steps for convenience in README and TUTORIAL.
189. [x] Roll back to earlier step outputs when experiments go wrong.
   - [x] Outline design for Roll back to earlier step outputs when experiments go wrong.
   - [x] Implement Roll back to earlier step outputs when experiments go wrong with CPU/GPU support.
   - [x] Add tests validating Roll back to earlier step outputs when experiments go wrong.
   - [x] Document Roll back to earlier step outputs when experiments go wrong in README and TUTORIAL.
190. [x] Integrate hyperparameter search that plugs directly into the pipeline engine.
   - [x] Outline design for Integrate hyperparameter search that plugs directly into the pipeline engine.
   - [x] Implement Integrate hyperparameter search that plugs directly into the pipeline engine with CPU/GPU support.
   - [x] Add tests validating Integrate hyperparameter search that plugs directly into the pipeline engine.
   - [x] Document Integrate hyperparameter search that plugs directly into the pipeline engine in README and TUTORIAL.
191. [ ] Schedule individual steps on remote hardware tiers seamlessly.
   - [ ] Outline design for Schedule individual steps on remote hardware tiers seamlessly.
   - [ ] Implement Schedule individual steps on remote hardware tiers seamlessly with CPU/GPU support.
   - [ ] Add tests validating Schedule individual steps on remote hardware tiers seamlessly.
   - [ ] Document Schedule individual steps on remote hardware tiers seamlessly in README and TUTORIAL.
192. [ ] Distribute dataset shards across parallel pipelines.
   - [ ] Outline design for Distribute dataset shards across parallel pipelines.
   - [ ] Implement Distribute dataset shards across parallel pipelines with CPU/GPU support.
   - [ ] Add tests validating Distribute dataset shards across parallel pipelines.
   - [ ] Document Distribute dataset shards across parallel pipelines in README and TUTORIAL.
193. [ ] Estimate resource needs ahead of execution to inform the memory manager.
   - [ ] Outline design for Estimate resource needs ahead of execution to inform the memory manager.
   - [ ] Implement Estimate resource needs ahead of execution to inform the memory manager with CPU/GPU support.
   - [ ] Add tests validating Estimate resource needs ahead of execution to inform the memory manager.
   - [ ] Document Estimate resource needs ahead of execution to inform the memory manager in README and TUTORIAL.
194. [ ] Save run profiles capturing the exact execution order.
   - [ ] Outline design for Save run profiles capturing the exact execution order.
   - [ ] Implement Save run profiles capturing the exact execution order with CPU/GPU support.
   - [ ] Add tests validating Save run profiles capturing the exact execution order.
   - [ ] Document Save run profiles capturing the exact execution order in README and TUTORIAL.
195. [ ] Edit pipeline definitions interactively through the GUI.
   - [ ] Outline design for Edit pipeline definitions interactively through the GUI.
   - [ ] Implement Edit pipeline definitions interactively through the GUI with CPU/GPU support.
   - [ ] Add tests validating Edit pipeline definitions interactively through the GUI.
   - [ ] Document Edit pipeline definitions interactively through the GUI in README and TUTORIAL.
196. [ ] Relay dataset events to pipeline notifications.
   - [ ] Outline design for Relay dataset events to pipeline notifications.
   - [ ] Implement Relay dataset events to pipeline notifications with CPU/GPU support.
   - [ ] Add tests validating Relay dataset events to pipeline notifications.
   - [ ] Document Relay dataset events to pipeline notifications in README and TUTORIAL.
197. [ ] Update Neuronenblitz models automatically when datasets change.
   - [ ] Outline design for Update Neuronenblitz models automatically when datasets change.
   - [ ] Implement Update Neuronenblitz models automatically when datasets change with CPU/GPU support.
   - [ ] Add tests validating Update Neuronenblitz models automatically when datasets change.
   - [ ] Document Update Neuronenblitz models automatically when datasets change in README and TUTORIAL.
198. [ ] Provide built-in cross-validation loops using deterministic dataset splits.
   - [ ] Outline design for Provide built-in cross-validation loops using deterministic dataset splits.
   - [ ] Implement Provide built-in cross-validation loops using deterministic dataset splits with CPU/GPU support.
   - [ ] Add tests validating Provide built-in cross-validation loops using deterministic dataset splits.
   - [ ] Document Provide built-in cross-validation loops using deterministic dataset splits in README and TUTORIAL.
199. [ ] Serve models through the web API directly from a pipeline step.
   - [ ] Outline design for Serve models through the web API directly from a pipeline step.
   - [ ] Implement Serve models through the web API directly from a pipeline step with CPU/GPU support.
   - [ ] Add tests validating Serve models through the web API directly from a pipeline step.
   - [ ] Document Serve models through the web API directly from a pipeline step in README and TUTORIAL.
200. [x] Benchmark pipeline steps using the core micro-benchmark tool.
201. [ ] Reorder steps dynamically based on dependency resolution.
   - [ ] Outline design for Reorder steps dynamically based on dependency resolution.
   - [ ] Implement Reorder steps dynamically based on dependency resolution with CPU/GPU support.
   - [ ] Add tests validating Reorder steps dynamically based on dependency resolution.
   - [ ] Document Reorder steps dynamically based on dependency resolution in README and TUTORIAL.
202. [x] Broadcast pipeline progress to the Global Workspace plugin.
203. [x] Route step logs to the metrics visualiser for real-time viewing.
204. [x] Diff pipeline configurations to track changes between runs.
205. [x] Stream logs from each step into the GUI console.
206. [x] Pre-allocate resources via the memory pool before executing steps.
207. [x] Freeze and defrost steps without removing them from the pipeline.
208. [ ] Run pipeline sections in isolated processes for fault tolerance.
   - [ ] Outline design for Run pipeline sections in isolated processes for fault tolerance.
   - [ ] Implement Run pipeline sections in isolated processes for fault tolerance with CPU/GPU support.
   - [ ] Add tests validating Run pipeline sections in isolated processes for fault tolerance.
   - [ ] Document Run pipeline sections in isolated processes for fault tolerance in README and TUTORIAL.
209. [ ] Connect with remote wanderers for asynchronous exploration phases.
   - [ ] Outline design for Connect with remote wanderers for asynchronous exploration phases.
   - [ ] Implement Connect with remote wanderers for asynchronous exploration phases with CPU/GPU support.
   - [ ] Add tests validating Connect with remote wanderers for asynchronous exploration phases.
   - [ ] Document Connect with remote wanderers for asynchronous exploration phases in README and TUTORIAL.
210. [ ] Secure pipeline data flow by integrating dataset encryption routines.
   - [ ] Outline design for Secure pipeline data flow by integrating dataset encryption routines.
   - [ ] Implement Secure pipeline data flow by integrating dataset encryption routines with CPU/GPU support.
   - [ ] Add tests validating Secure pipeline data flow by integrating dataset encryption routines.
   - [ ] Document Secure pipeline data flow by integrating dataset encryption routines in README and TUTORIAL.
211. [ ] Route memory allocations through the memory pool for every operation.
   - [ ] Outline design for Route memory allocations through the memory pool for every operation.
   - [ ] Implement Route memory allocations through the memory pool for every operation with CPU/GPU support.
   - [ ] Add tests validating Route memory allocations through the memory pool for every operation.
   - [ ] Document Route memory allocations through the memory pool for every operation in README and TUTORIAL.
212. [x] Provide a CLI wrapper so pipelines can run without writing Python code.
213. [x] Detect GPU availability and adapt pipeline behaviour automatically.
214. [x] Persist vocabulary mappings for reuse across multiple runs.
215. [ ] Train directly from streamed dataset shards loaded via pipeline steps.
   - [ ] Outline design for Train directly from streamed dataset shards loaded via pipeline steps.
   - [ ] Implement Train directly from streamed dataset shards loaded via pipeline steps with CPU/GPU support.
   - [ ] Add tests validating Train directly from streamed dataset shards loaded via pipeline steps.
   - [ ] Document Train directly from streamed dataset shards loaded via pipeline steps in README and TUTORIAL.
216. [ ] Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements.
   - [ ] Outline design for Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements.
   - [ ] Implement Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements with CPU/GPU support.
   - [ ] Add tests validating Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements.
   - [ ] Document Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements in README and TUTORIAL.
217. [ ] Support streaming dataset shards during Neuronenblitz training to keep the model responsive.
   - [ ] Outline design for Support streaming dataset shards during Neuronenblitz training to keep the model responsive.
   - [ ] Implement Support streaming dataset shards during Neuronenblitz training to keep the model responsive with CPU/GPU support.
   - [ ] Add tests validating Support streaming dataset shards during Neuronenblitz training to keep the model responsive.
   - [ ] Document Support streaming dataset shards during Neuronenblitz training to keep the model responsive in README and TUTORIAL.
218. [ ] Allow learning modules to be swapped in and out through a plugin interface.
   - [ ] Outline design for Allow learning modules to be swapped in and out through a plugin interface.
   - [ ] Implement Allow learning modules to be swapped in and out through a plugin interface with CPU/GPU support.
   - [ ] Add tests validating Allow learning modules to be swapped in and out through a plugin interface.
   - [ ] Document Allow learning modules to be swapped in and out through a plugin interface in README and TUTORIAL.
219. [ ] Use Global Workspace events to guide dynamic attention gating.
   - [ ] Outline design for Use Global Workspace events to guide dynamic attention gating.
   - [ ] Implement Use Global Workspace events to guide dynamic attention gating with CPU/GPU support.
   - [ ] Add tests validating Use Global Workspace events to guide dynamic attention gating.
   - [ ] Document Use Global Workspace events to guide dynamic attention gating in README and TUTORIAL.
220. [ ] Provide a reinforcement learning loop coordinated by pipeline scheduling.
   - [ ] Outline design for Provide a reinforcement learning loop coordinated by pipeline scheduling.
   - [ ] Implement Provide a reinforcement learning loop coordinated by pipeline scheduling with CPU/GPU support.
   - [ ] Add tests validating Provide a reinforcement learning loop coordinated by pipeline scheduling.
   - [ ] Document Provide a reinforcement learning loop coordinated by pipeline scheduling in README and TUTORIAL.
221. [ ] Offload wandering to remote hardware using Marble Core utilities.
   - [ ] Outline design for Offload wandering to remote hardware using Marble Core utilities.
   - [ ] Implement Offload wandering to remote hardware using Marble Core utilities with CPU/GPU support.
   - [ ] Add tests validating Offload wandering to remote hardware using Marble Core utilities.
   - [ ] Document Offload wandering to remote hardware using Marble Core utilities in README and TUTORIAL.
222. [x] Optimise memory usage by sharing dataset caches with the memory pool.
223. [ ] Accumulate gradients asynchronously in line with pipeline scheduling.
   - [ ] Outline design for Accumulate gradients asynchronously in line with pipeline scheduling.
   - [ ] Implement Accumulate gradients asynchronously in line with pipeline scheduling with CPU/GPU support.
   - [ ] Add tests validating Accumulate gradients asynchronously in line with pipeline scheduling.
   - [ ] Document Accumulate gradients asynchronously in line with pipeline scheduling in README and TUTORIAL.
224. [ ] Inspect neural pathways interactively via the GUI.
   - [ ] Outline design for Inspect neural pathways interactively via the GUI.
   - [ ] Implement Inspect neural pathways interactively via the GUI with CPU/GPU support.
   - [ ] Add tests validating Inspect neural pathways interactively via the GUI.
   - [ ] Document Inspect neural pathways interactively via the GUI in README and TUTORIAL.
225. [ ] Register custom loss modules through the plugin system.
   - [ ] Outline design for Register custom loss modules through the plugin system.
   - [ ] Implement Register custom loss modules through the plugin system with CPU/GPU support.
   - [ ] Add tests validating Register custom loss modules through the plugin system.
   - [ ] Document Register custom loss modules through the plugin system in README and TUTORIAL.
226. [ ] Transfer knowledge between models using dataset serialisation features.
   - [ ] Outline design for Transfer knowledge between models using dataset serialisation features.
   - [ ] Implement Transfer knowledge between models using dataset serialisation features with CPU/GPU support.
   - [ ] Add tests validating Transfer knowledge between models using dataset serialisation features.
   - [ ] Document Transfer knowledge between models using dataset serialisation features in README and TUTORIAL.
227. [ ] Refresh vocabulary encodings mid-training when datasets evolve.
   - [ ] Outline design for Refresh vocabulary encodings mid-training when datasets evolve.
   - [ ] Implement Refresh vocabulary encodings mid-training when datasets evolve with CPU/GPU support.
   - [ ] Add tests validating Refresh vocabulary encodings mid-training when datasets evolve.
   - [ ] Document Refresh vocabulary encodings mid-training when datasets evolve in README and TUTORIAL.
228. [ ] Evaluate models remotely using the pipeline inference plugin.
   - [ ] Outline design for Evaluate models remotely using the pipeline inference plugin.
   - [ ] Implement Evaluate models remotely using the pipeline inference plugin with CPU/GPU support.
   - [ ] Add tests validating Evaluate models remotely using the pipeline inference plugin.
   - [ ] Document Evaluate models remotely using the pipeline inference plugin in README and TUTORIAL.
229. [ ] Plan actions hierarchically using Global Workspace goals.
   - [ ] Outline design for Plan actions hierarchically using Global Workspace goals.
   - [ ] Implement Plan actions hierarchically using Global Workspace goals with CPU/GPU support.
   - [ ] Add tests validating Plan actions hierarchically using Global Workspace goals.
   - [ ] Document Plan actions hierarchically using Global Workspace goals in README and TUTORIAL.
230. [ ] Adjust curricula automatically based on dataset history.
   - [ ] Outline design for Adjust curricula automatically based on dataset history.
   - [ ] Implement Adjust curricula automatically based on dataset history with CPU/GPU support.
   - [ ] Add tests validating Adjust curricula automatically based on dataset history.
   - [ ] Document Adjust curricula automatically based on dataset history in README and TUTORIAL.
231. [ ] Update plasticity parameters from dataset augmentation events.
   - [ ] Outline design for Update plasticity parameters from dataset augmentation events.
   - [ ] Implement Update plasticity parameters from dataset augmentation events with CPU/GPU support.
   - [ ] Add tests validating Update plasticity parameters from dataset augmentation events.
   - [ ] Document Update plasticity parameters from dataset augmentation events in README and TUTORIAL.
232. [ ] Allocate weights from the GPU memory pool for efficient updates.
   - [ ] Outline design for Allocate weights from the GPU memory pool for efficient updates.
   - [ ] Implement Allocate weights from the GPU memory pool for efficient updates with CPU/GPU support.
   - [ ] Add tests validating Allocate weights from the GPU memory pool for efficient updates.
   - [ ] Document Allocate weights from the GPU memory pool for efficient updates in README and TUTORIAL.
233. [ ] Step through the pipeline debugger during training runs.
   - [ ] Outline design for Step through the pipeline debugger during training runs.
   - [ ] Implement Step through the pipeline debugger during training runs with CPU/GPU support.
   - [ ] Add tests validating Step through the pipeline debugger during training runs.
   - [ ] Document Step through the pipeline debugger during training runs in README and TUTORIAL.
234. [ ] Wander asynchronously while prefetching dataset shards.
   - [ ] Outline design for Wander asynchronously while prefetching dataset shards.
   - [ ] Implement Wander asynchronously while prefetching dataset shards with CPU/GPU support.
   - [ ] Add tests validating Wander asynchronously while prefetching dataset shards.
   - [ ] Document Wander asynchronously while prefetching dataset shards in README and TUTORIAL.
235. [ ] Include a self-supervised module that consumes augmented data.
   - [ ] Outline design for Include a self-supervised module that consumes augmented data.
   - [ ] Implement Include a self-supervised module that consumes augmented data with CPU/GPU support.
   - [ ] Add tests validating Include a self-supervised module that consumes augmented data.
   - [ ] Document Include a self-supervised module that consumes augmented data in README and TUTORIAL.
236. [ ] Share memory buffers across nodes with the remote memory pool.
   - [ ] Outline design for Share memory buffers across nodes with the remote memory pool.
   - [ ] Implement Share memory buffers across nodes with the remote memory pool with CPU/GPU support.
   - [ ] Add tests validating Share memory buffers across nodes with the remote memory pool.
   - [ ] Document Share memory buffers across nodes with the remote memory pool in README and TUTORIAL.
237. [ ] Load synapse types dynamically via pipeline plugins.
   - [ ] Outline design for Load synapse types dynamically via pipeline plugins.
   - [ ] Implement Load synapse types dynamically via pipeline plugins with CPU/GPU support.
   - [ ] Add tests validating Load synapse types dynamically via pipeline plugins.
   - [ ] Document Load synapse types dynamically via pipeline plugins in README and TUTORIAL.
238. [ ] Checkpoint models in a universal format understood by the pipeline.
   - [ ] Outline design for Checkpoint models in a universal format understood by the pipeline.
   - [ ] Implement Checkpoint models in a universal format understood by the pipeline with CPU/GPU support.
   - [ ] Add tests validating Checkpoint models in a universal format understood by the pipeline.
   - [ ] Document Checkpoint models in a universal format understood by the pipeline in README and TUTORIAL.
239. [ ] Inject gradient noise derived from dataset noise augmentation.
   - [ ] Outline design for Inject gradient noise derived from dataset noise augmentation.
   - [ ] Implement Inject gradient noise derived from dataset noise augmentation with CPU/GPU support.
   - [ ] Add tests validating Inject gradient noise derived from dataset noise augmentation.
   - [ ] Document Inject gradient noise derived from dataset noise augmentation in README and TUTORIAL.
240. [ ] Prune routes when deduplication removes redundant data.
   - [ ] Outline design for Prune routes when deduplication removes redundant data.
   - [ ] Implement Prune routes when deduplication removes redundant data with CPU/GPU support.
   - [ ] Add tests validating Prune routes when deduplication removes redundant data.
   - [ ] Document Prune routes when deduplication removes redundant data in README and TUTORIAL.
241. [ ] Explore dataset samples in the interactive browser during training.
   - [ ] Outline design for Explore dataset samples in the interactive browser during training.
   - [ ] Implement Explore dataset samples in the interactive browser during training with CPU/GPU support.
   - [ ] Add tests validating Explore dataset samples in the interactive browser during training.
   - [ ] Document Explore dataset samples in the interactive browser during training in README and TUTORIAL.
242. [ ] Warm-start models from partial pipeline outputs.
   - [ ] Outline design for Warm-start models from partial pipeline outputs.
   - [ ] Implement Warm-start models from partial pipeline outputs with CPU/GPU support.
   - [ ] Add tests validating Warm-start models from partial pipeline outputs.
   - [ ] Document Warm-start models from partial pipeline outputs in README and TUTORIAL.
243. [ ] Adapt learning rates using events from the memory manager.
   - [ ] Outline design for Adapt learning rates using events from the memory manager.
   - [ ] Implement Adapt learning rates using events from the memory manager with CPU/GPU support.
   - [ ] Add tests validating Adapt learning rates using events from the memory manager.
   - [ ] Document Adapt learning rates using events from the memory manager in README and TUTORIAL.
244. [ ] Decrypt encrypted data on the fly during training.
   - [ ] Outline design for Decrypt encrypted data on the fly during training.
   - [ ] Implement Decrypt encrypted data on the fly during training with CPU/GPU support.
   - [ ] Add tests validating Decrypt encrypted data on the fly during training.
   - [ ] Document Decrypt encrypted data on the fly during training in README and TUTORIAL.
245. [ ] Aggregate gradients remotely using distributed helpers.
   - [ ] Outline design for Aggregate gradients remotely using distributed helpers.
   - [ ] Implement Aggregate gradients remotely using distributed helpers with CPU/GPU support.
   - [ ] Add tests validating Aggregate gradients remotely using distributed helpers.
   - [ ] Document Aggregate gradients remotely using distributed helpers in README and TUTORIAL.
246. [ ] Retrieve activations using approximate nearest neighbour search.
   - [ ] Outline design for Retrieve activations using approximate nearest neighbour search.
   - [ ] Implement Retrieve activations using approximate nearest neighbour search with CPU/GPU support.
   - [ ] Add tests validating Retrieve activations using approximate nearest neighbour search.
   - [ ] Document Retrieve activations using approximate nearest neighbour search in README and TUTORIAL.
247. [ ] Gating mechanisms use dataset tags for context.
   - [ ] Outline design for Gating mechanisms use dataset tags for context.
   - [ ] Implement Gating mechanisms use dataset tags for context with CPU/GPU support.
   - [ ] Add tests validating Gating mechanisms use dataset tags for context.
   - [ ] Document Gating mechanisms use dataset tags for context in README and TUTORIAL.
248. [ ] Reinitialize parts of the network when datasets are patched.
   - [ ] Outline design for Reinitialize parts of the network when datasets are patched.
   - [ ] Implement Reinitialize parts of the network when datasets are patched with CPU/GPU support.
   - [ ] Add tests validating Reinitialize parts of the network when datasets are patched.
   - [ ] Document Reinitialize parts of the network when datasets are patched in README and TUTORIAL.
249. [ ] Evaluate intermediate models using generated validation sets.
   - [ ] Outline design for Evaluate intermediate models using generated validation sets.
   - [ ] Implement Evaluate intermediate models using generated validation sets with CPU/GPU support.
   - [ ] Add tests validating Evaluate intermediate models using generated validation sets.
   - [ ] Document Evaluate intermediate models using generated validation sets in README and TUTORIAL.
250. [ ] Add runtime extension for registering new neuron types.
   - [ ] Outline design for Add runtime extension for registering new neuron types.
   - [ ] Implement Add runtime extension for registering new neuron types with CPU/GPU support.
   - [ ] Add tests validating Add runtime extension for registering new neuron types.
   - [ ] Document Add runtime extension for registering new neuron types in README and TUTORIAL.
251. [ ] Communicate between models using Global Workspace broadcast.
   - [ ] Outline design for Communicate between models using Global Workspace broadcast.
   - [ ] Implement Communicate between models using Global Workspace broadcast with CPU/GPU support.
   - [ ] Add tests validating Communicate between models using Global Workspace broadcast.
   - [ ] Document Communicate between models using Global Workspace broadcast in README and TUTORIAL.
252. [ ] Enable dropout gating informed by dataset quality hooks.
   - [ ] Outline design for Enable dropout gating informed by dataset quality hooks.
   - [ ] Implement Enable dropout gating informed by dataset quality hooks with CPU/GPU support.
   - [ ] Add tests validating Enable dropout gating informed by dataset quality hooks.
   - [ ] Document Enable dropout gating informed by dataset quality hooks in README and TUTORIAL.
253. [ ] Derive reinforcement signals from pipeline event logs.
   - [ ] Outline design for Derive reinforcement signals from pipeline event logs.
   - [ ] Implement Derive reinforcement signals from pipeline event logs with CPU/GPU support.
   - [ ] Add tests validating Derive reinforcement signals from pipeline event logs.
   - [ ] Document Derive reinforcement signals from pipeline event logs in README and TUTORIAL.
254. [ ] Mix offline and online learning phases based on dataset modifications.
   - [ ] Outline design for Mix offline and online learning phases based on dataset modifications.
   - [ ] Implement Mix offline and online learning phases based on dataset modifications with CPU/GPU support.
   - [ ] Add tests validating Mix offline and online learning phases based on dataset modifications.
   - [ ] Document Mix offline and online learning phases based on dataset modifications in README and TUTORIAL.
255. [ ] Integrate with hyperparameter sweeps to evaluate multiple runs.
   - [ ] Outline design for Integrate with hyperparameter sweeps to evaluate multiple runs.
   - [ ] Implement Integrate with hyperparameter sweeps to evaluate multiple runs with CPU/GPU support.
   - [ ] Add tests validating Integrate with hyperparameter sweeps to evaluate multiple runs.
   - [ ] Document Integrate with hyperparameter sweeps to evaluate multiple runs in README and TUTORIAL.
256. [ ] Use the memory pool for gradient buffer management.
   - [ ] Outline design for Use the memory pool for gradient buffer management.
   - [ ] Implement Use the memory pool for gradient buffer management with CPU/GPU support.
   - [ ] Add tests validating Use the memory pool for gradient buffer management.
   - [ ] Document Use the memory pool for gradient buffer management in README and TUTORIAL.
257. [ ] Route decisions based on hierarchical dataset tags.
   - [ ] Outline design for Route decisions based on hierarchical dataset tags.
   - [ ] Implement Route decisions based on hierarchical dataset tags with CPU/GPU support.
   - [ ] Add tests validating Route decisions based on hierarchical dataset tags.
   - [ ] Document Route decisions based on hierarchical dataset tags in README and TUTORIAL.
258. [ ] Cache activations for repeated passes over dataset shards.
   - [ ] Outline design for Cache activations for repeated passes over dataset shards.
   - [ ] Implement Cache activations for repeated passes over dataset shards with CPU/GPU support.
   - [ ] Add tests validating Cache activations for repeated passes over dataset shards.
   - [ ] Document Cache activations for repeated passes over dataset shards in README and TUTORIAL.
259. [x] Send training events to the metrics visualiser.
    - [x] Add log_event method to MetricsVisualizer.
    - [x] Emit epoch_start and epoch_end in Brain.train.
    - [x] Display events in GUI console.
260. [ ] Ensure encrypted datasets remain private throughout training.
   - [ ] Outline design for Ensure encrypted datasets remain private throughout training.
   - [ ] Implement Ensure encrypted datasets remain private throughout training with CPU/GPU support.
   - [ ] Add tests validating Ensure encrypted datasets remain private throughout training.
   - [ ] Document Ensure encrypted datasets remain private throughout training in README and TUTORIAL.
261. [ ] Replicate networks across nodes using remote hardware plugins.
   - [ ] Outline design for Replicate networks across nodes using remote hardware plugins.
   - [ ] Implement Replicate networks across nodes using remote hardware plugins with CPU/GPU support.
   - [ ] Add tests validating Replicate networks across nodes using remote hardware plugins.
   - [ ] Document Replicate networks across nodes using remote hardware plugins in README and TUTORIAL.
262. [ ] Quickly evaluate models via pipeline checkpoint restore.
   - [ ] Outline design for Quickly evaluate models via pipeline checkpoint restore.
   - [ ] Implement Quickly evaluate models via pipeline checkpoint restore with CPU/GPU support.
   - [ ] Add tests validating Quickly evaluate models via pipeline checkpoint restore.
   - [ ] Document Quickly evaluate models via pipeline checkpoint restore in README and TUTORIAL.
263. [ ] Roll back neuron states when dataset audits fail.
   - [ ] Outline design for Roll back neuron states when dataset audits fail.
   - [ ] Implement Roll back neuron states when dataset audits fail with CPU/GPU support.
   - [ ] Add tests validating Roll back neuron states when dataset audits fail.
   - [ ] Document Roll back neuron states when dataset audits fail in README and TUTORIAL.
264. [ ] Adapt gating when vocabulary updates occur.
   - [ ] Outline design for Adapt gating when vocabulary updates occur.
   - [ ] Implement Adapt gating when vocabulary updates occur with CPU/GPU support.
   - [ ] Add tests validating Adapt gating when vocabulary updates occur.
   - [ ] Document Adapt gating when vocabulary updates occur in README and TUTORIAL.
265. [ ] Manage experiments across runs using pipeline and dataset versions.
   - [ ] Outline design for Manage experiments across runs using pipeline and dataset versions.
   - [ ] Implement Manage experiments across runs using pipeline and dataset versions with CPU/GPU support.
   - [ ] Add tests validating Manage experiments across runs using pipeline and dataset versions.
   - [ ] Document Manage experiments across runs using pipeline and dataset versions in README and TUTORIAL.
266. [ ] Build a distributed memory pool so datasets are shared across nodes.
   - [ ] Outline design for Build a distributed memory pool so datasets are shared across nodes.
   - [ ] Implement Build a distributed memory pool so datasets are shared across nodes with CPU/GPU support.
   - [ ] Add tests validating Build a distributed memory pool so datasets are shared across nodes.
   - [ ] Document Build a distributed memory pool so datasets are shared across nodes in README and TUTORIAL.
267. [ ] Accelerate bit tensor operations on GPU for faster dataset processing.
   - [ ] Outline design for Accelerate bit tensor operations on GPU for faster dataset processing.
   - [ ] Implement Accelerate bit tensor operations on GPU for faster dataset processing with CPU/GPU support.
   - [ ] Add tests validating Accelerate bit tensor operations on GPU for faster dataset processing.
   - [ ] Document Accelerate bit tensor operations on GPU for faster dataset processing in README and TUTORIAL.
268. [ ] Provide a remote procedure interface to run pipeline steps asynchronously.
   - [ ] Outline design for Provide a remote procedure interface to run pipeline steps asynchronously.
   - [ ] Implement Provide a remote procedure interface to run pipeline steps asynchronously with CPU/GPU support.
   - [ ] Add tests validating Provide a remote procedure interface to run pipeline steps asynchronously.
   - [ ] Document Provide a remote procedure interface to run pipeline steps asynchronously in README and TUTORIAL.
269. [ ] Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules.
   - [ ] Outline design for Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules.
   - [ ] Implement Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules with CPU/GPU support.
   - [ ] Add tests validating Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules.
   - [ ] Document Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules in README and TUTORIAL.
270. [ ] Offer encryption utilities supporting secure dataset storage.
   - [ ] Outline design for Offer encryption utilities supporting secure dataset storage.
   - [ ] Implement Offer encryption utilities supporting secure dataset storage with CPU/GPU support.
   - [ ] Add tests validating Offer encryption utilities supporting secure dataset storage.
   - [ ] Document Offer encryption utilities supporting secure dataset storage in README and TUTORIAL.
271. [ ] Handle memory-mapped tensors for large streaming datasets.
   - [ ] Outline design for Handle memory-mapped tensors for large streaming datasets.
   - [ ] Implement Handle memory-mapped tensors for large streaming datasets with CPU/GPU support.
   - [ ] Add tests validating Handle memory-mapped tensors for large streaming datasets.
   - [ ] Document Handle memory-mapped tensors for large streaming datasets in README and TUTORIAL.
272. [ ] Load configuration hierarchies merging dataset and pipeline settings.
   - [ ] Outline design for Load configuration hierarchies merging dataset and pipeline settings.
   - [ ] Implement Load configuration hierarchies merging dataset and pipeline settings with CPU/GPU support.
   - [ ] Add tests validating Load configuration hierarchies merging dataset and pipeline settings.
   - [ ] Document Load configuration hierarchies merging dataset and pipeline settings in README and TUTORIAL.
273. [ ] Add backpressure-aware message passing for streaming data.
   - [ ] Outline design for Add backpressure-aware message passing for streaming data.
   - [ ] Implement Add backpressure-aware message passing for streaming data with CPU/GPU support.
   - [ ] Add tests validating Add backpressure-aware message passing for streaming data.
   - [ ] Document Add backpressure-aware message passing for streaming data in README and TUTORIAL.
274. [ ] Create a cross-process checkpoint manager accessible from pipelines.
   - [ ] Outline design for Create a cross-process checkpoint manager accessible from pipelines.
   - [ ] Implement Create a cross-process checkpoint manager accessible from pipelines with CPU/GPU support.
   - [ ] Add tests validating Create a cross-process checkpoint manager accessible from pipelines.
   - [ ] Document Create a cross-process checkpoint manager accessible from pipelines in README and TUTORIAL.
275. [ ] Dynamically load neuron and synapse plugins at runtime.
   - [ ] Outline design for Dynamically load neuron and synapse plugins at runtime.
   - [ ] Implement Dynamically load neuron and synapse plugins at runtime with CPU/GPU support.
   - [ ] Add tests validating Dynamically load neuron and synapse plugins at runtime.
   - [ ] Document Dynamically load neuron and synapse plugins at runtime in README and TUTORIAL.
276. [ ] Debug message passing flows in real time through GUI tools.
   - [ ] Outline design for Debug message passing flows in real time through GUI tools.
   - [ ] Implement Debug message passing flows in real time through GUI tools with CPU/GPU support.
   - [ ] Add tests validating Debug message passing flows in real time through GUI tools.
   - [ ] Document Debug message passing flows in real time through GUI tools in README and TUTORIAL.
277. [ ] Provide a remote scheduler for distributed pipeline execution.
   - [ ] Outline design for Provide a remote scheduler for distributed pipeline execution.
   - [ ] Implement Provide a remote scheduler for distributed pipeline execution with CPU/GPU support.
   - [ ] Add tests validating Provide a remote scheduler for distributed pipeline execution.
   - [ ] Document Provide a remote scheduler for distributed pipeline execution in README and TUTORIAL.
278. [ ] Verify data integrity across modules with unified checksum tools.
   - [ ] Outline design for Verify data integrity across modules with unified checksum tools.
   - [ ] Implement Verify data integrity across modules with unified checksum tools with CPU/GPU support.
   - [ ] Add tests validating Verify data integrity across modules with unified checksum tools.
   - [ ] Document Verify data integrity across modules with unified checksum tools in README and TUTORIAL.
279. [ ] Serialise models and datasets through a cross-platform layer.
   - [ ] Outline design for Serialise models and datasets through a cross-platform layer.
   - [ ] Implement Serialise models and datasets through a cross-platform layer with CPU/GPU support.
   - [ ] Add tests validating Serialise models and datasets through a cross-platform layer.
   - [ ] Document Serialise models and datasets through a cross-platform layer in README and TUTORIAL.
280. [ ] Cache activation tensors that Neuronenblitz reuses during training.
   - [ ] Outline design for Cache activation tensors that Neuronenblitz reuses during training.
   - [ ] Implement Cache activation tensors that Neuronenblitz reuses during training with CPU/GPU support.
   - [ ] Add tests validating Cache activation tensors that Neuronenblitz reuses during training.
   - [ ] Document Cache activation tensors that Neuronenblitz reuses during training in README and TUTORIAL.
281. [ ] Securely store encryption keys via a dedicated security manager.
   - [ ] Outline design for Securely store encryption keys via a dedicated security manager.
   - [ ] Implement Securely store encryption keys via a dedicated security manager with CPU/GPU support.
   - [ ] Add tests validating Securely store encryption keys via a dedicated security manager.
   - [ ] Document Securely store encryption keys via a dedicated security manager in README and TUTORIAL.
282. [ ] Hot‑reload modules without stopping ongoing training.
   - [ ] Outline design for Hot‑reload modules without stopping ongoing training.
   - [ ] Implement Hot‑reload modules without stopping ongoing training with CPU/GPU support.
   - [ ] Add tests validating Hot‑reload modules without stopping ongoing training.
   - [ ] Document Hot‑reload modules without stopping ongoing training in README and TUTORIAL.
283. [ ] Manage GPU memory for dataset prefetching.
   - [ ] Outline design for Manage GPU memory for dataset prefetching.
   - [ ] Implement Manage GPU memory for dataset prefetching with CPU/GPU support.
   - [ ] Add tests validating Manage GPU memory for dataset prefetching.
   - [ ] Document Manage GPU memory for dataset prefetching in README and TUTORIAL.
284. [ ] Support asynchronous I/O for continuous dataset streaming.
   - [ ] Outline design for Support asynchronous I/O for continuous dataset streaming.
   - [ ] Implement Support asynchronous I/O for continuous dataset streaming with CPU/GPU support.
   - [ ] Add tests validating Support asynchronous I/O for continuous dataset streaming.
   - [ ] Document Support asynchronous I/O for continuous dataset streaming in README and TUTORIAL.
285. [ ] Emit resource events to notify components of memory pressure.
   - [ ] Outline design for Emit resource events to notify components of memory pressure.
   - [ ] Implement Emit resource events to notify components of memory pressure with CPU/GPU support.
   - [ ] Add tests validating Emit resource events to notify components of memory pressure.
   - [ ] Document Emit resource events to notify components of memory pressure in README and TUTORIAL.
286. [ ] Replicate memory pools across machines for distributed workloads.
   - [ ] Outline design for Replicate memory pools across machines for distributed workloads.
   - [ ] Implement Replicate memory pools across machines for distributed workloads with CPU/GPU support.
   - [ ] Add tests validating Replicate memory pools across machines for distributed workloads.
   - [ ] Document Replicate memory pools across machines for distributed workloads in README and TUTORIAL.
287. [x] Validate configurations globally before starting a pipeline.
    - [x] Implement validate_global_config function.
    - [x] Add schema checks for all sections.
288. [ ] Provide thread-safe APIs for parallel dataset transformations.
   - [ ] Outline design for Provide thread-safe APIs for parallel dataset transformations.
   - [ ] Implement Provide thread-safe APIs for parallel dataset transformations with CPU/GPU support.
   - [ ] Add tests validating Provide thread-safe APIs for parallel dataset transformations.
   - [ ] Document Provide thread-safe APIs for parallel dataset transformations in README and TUTORIAL.
289. [ ] Deduplicate repeated bit tensors across different datasets.
   - [ ] Outline design for Deduplicate repeated bit tensors across different datasets.
   - [ ] Implement Deduplicate repeated bit tensors across different datasets with CPU/GPU support.
   - [ ] Add tests validating Deduplicate repeated bit tensors across different datasets.
   - [ ] Document Deduplicate repeated bit tensors across different datasets in README and TUTORIAL.
290. [ ] Aggregate event logs from all modules into a unified stream.
   - [ ] Outline design for Aggregate event logs from all modules into a unified stream.
   - [ ] Implement Aggregate event logs from all modules into a unified stream with CPU/GPU support.
   - [ ] Add tests validating Aggregate event logs from all modules into a unified stream.
   - [ ] Document Aggregate event logs from all modules into a unified stream in README and TUTORIAL.
291. [ ] Update models with partial graph recompilation.
   - [ ] Outline design for Update models with partial graph recompilation.
   - [ ] Implement Update models with partial graph recompilation with CPU/GPU support.
   - [ ] Add tests validating Update models with partial graph recompilation.
   - [ ] Document Update models with partial graph recompilation in README and TUTORIAL.
292. [ ] Accelerate nearest neighbour search via specialised hardware plugins.
   - [ ] Outline design for Accelerate nearest neighbour search via specialised hardware plugins.
   - [ ] Implement Accelerate nearest neighbour search via specialised hardware plugins with CPU/GPU support.
   - [ ] Add tests validating Accelerate nearest neighbour search via specialised hardware plugins.
   - [ ] Document Accelerate nearest neighbour search via specialised hardware plugins in README and TUTORIAL.
293. [ ] Queue wander jobs reliably even if a process crashes.
   - [ ] Outline design for Queue wander jobs reliably even if a process crashes.
   - [ ] Implement Queue wander jobs reliably even if a process crashes with CPU/GPU support.
   - [ ] Add tests validating Queue wander jobs reliably even if a process crashes.
   - [ ] Document Queue wander jobs reliably even if a process crashes in README and TUTORIAL.
294. [ ] Synchronise checkpoints automatically across nodes.
   - [ ] Outline design for Synchronise checkpoints automatically across nodes.
   - [ ] Implement Synchronise checkpoints automatically across nodes with CPU/GPU support.
   - [ ] Add tests validating Synchronise checkpoints automatically across nodes.
   - [ ] Document Synchronise checkpoints automatically across nodes in README and TUTORIAL.
295. [ ] Schedule steps using a GPU-aware dispatcher.
   - [ ] Outline design for Schedule steps using a GPU-aware dispatcher.
   - [ ] Implement Schedule steps using a GPU-aware dispatcher with CPU/GPU support.
   - [ ] Add tests validating Schedule steps using a GPU-aware dispatcher.
   - [ ] Document Schedule steps using a GPU-aware dispatcher in README and TUTORIAL.
296. [ ] Prune unused neurons according to dataset prune events.
   - [ ] Outline design for Prune unused neurons according to dataset prune events.
   - [ ] Implement Prune unused neurons according to dataset prune events with CPU/GPU support.
   - [ ] Add tests validating Prune unused neurons according to dataset prune events.
   - [ ] Document Prune unused neurons according to dataset prune events in README and TUTORIAL.
297. [ ] Propagate context from pipeline events into core behaviours.
   - [ ] Outline design for Propagate context from pipeline events into core behaviours.
   - [ ] Implement Propagate context from pipeline events into core behaviours with CPU/GPU support.
   - [ ] Add tests validating Propagate context from pipeline events into core behaviours.
   - [ ] Document Propagate context from pipeline events into core behaviours in README and TUTORIAL.
298. [ ] Visualise core operations in a built-in GUI server.
   - [ ] Outline design for Visualise core operations in a built-in GUI server.
   - [ ] Implement Visualise core operations in a built-in GUI server with CPU/GPU support.
   - [ ] Add tests validating Visualise core operations in a built-in GUI server.
   - [ ] Document Visualise core operations in a built-in GUI server in README and TUTORIAL.
299. [x] Cache downloaded datasets at the network level.
    - [x] Implement DatasetCacheServer to share downloads.
    - [x] Modify load_dataset to query remote cache.
300. [ ] Orchestrate cross-validation using core utilities and dataset splits.
    - [ ] Implement K-fold dataset splitter producing training and validation subsets.
    - [ ] Add cross-validation runner to training pipeline.
    - [ ] Expose CLI and YAML options to select number of folds.
    - [ ] Add tests verifying aggregated metrics across folds.
301. [ ] Spawn remote workers to handle dataset transformations.
    - [ ] Define worker API for preprocessing jobs.
    - [ ] Implement remote worker pool using RPC.
    - [ ] Dispatch pipeline transformations to worker pool.
    - [ ] Provide tests simulating worker failures and recoveries.
302. [ ] Serialise pipeline definitions through a portable format.
    - [ ] Define JSON schema capturing pipeline steps and parameters.
    - [ ] Implement serializer and deserializer utilities.
    - [ ] Support loading pipeline definitions from file or CLI.
    - [ ] Add round-trip serialization tests.
303. [ ] Aggregate events and feed them to the metrics visualiser.
    - [ ] Define event bus API for pipeline components.
    - [ ] Implement aggregator collecting events across processes.
    - [ ] Integrate aggregator output with metrics dashboard.
    - [ ] Test event flow with mock components.
304. [ ] Protect remote memory operations with encryption utilities.
    - [ ] Add symmetric encryption helper for memory transfers.
    - [ ] Store keys in configuration with rotation support.
    - [ ] Encrypt and decrypt data during remote operations.
    - [ ] Write tests verifying encrypted transfers.
305. [ ] Balance CPU and GPU resources for dataset handling.
    - [ ] Profile pipeline steps to measure CPU/GPU load.
    - [ ] Add scheduler assigning operations to appropriate device.
    - [ ] Expose YAML parameters to tune balancing strategy.
    - [ ] Test pipeline under CPU-only and GPU modes.
306. [x] Provide a dedicated test harness for bit tensor functions.
    - [x] Create helper class for generating datasets.
    - [x] Integrate harness with existing tests.
307. [ ] Enforce memory quotas per pipeline step.
    - [ ] Introduce configuration parameter for per-step memory limit.
    - [ ] Instrument steps to track allocated memory.
    - [ ] Abort or queue step when quota exceeded.
    - [ ] Add tests simulating quota breaches.
308. [ ] Precompile compute graphs to accelerate training.
    - [ ] Implement graph caching for repeated computations.
    - [ ] Add precompilation phase in training initialization.
    - [ ] Provide CLI flag to toggle precompilation.
    - [ ] Benchmark speed improvements on sample models.
309. [ ] Offer multi-step undo for dataset modifications via core services.
    - [ ] Track modification history with unique IDs.
    - [ ] Implement undo stack supporting multiple levels.
    - [ ] Expose CLI and GUI controls to revert operations.
    - [ ] Add tests covering undo and redo logic.
310. [ ] Update remote datasets incrementally during long experiments.
    - [ ] Detect dataset changes and compute delta patches.
    - [ ] Sync remote storage with incremental updates.
    - [ ] Provide progress reporting for each sync.
    - [ ] Add tests verifying no data loss.
311. [ ] Use a plugin-based scheduler for asynchronous tasks.
    - [ ] Define scheduler plugin interface.
    - [ ] Implement default thread and asyncio scheduler plugins.
    - [ ] Allow selection via configuration or CLI.
    - [ ] Test plugin lifecycle and task dispatch.
312. [ ] Standardise event formats so dataset and pipeline logs are compatible.
    - [ ] Define unified event schema.
    - [ ] Update logging emitters to output schema-compliant events.
    - [ ] Provide converter for legacy log formats.
    - [ ] Add tests validating event schema.
313. [ ] Supply CLI tools to manage dataset versions across platforms.
    - [ ] Implement `dataset version` CLI commands (list, create, switch).
    - [ ] Support both local and remote registries.
    - [ ] Integrate version info with pipeline loader.
    - [ ] Add tests for version switching.
314. [ ] Integrate with the Global Workspace to monitor overall system state.
    - [x] Expose pipeline metrics to Global Workspace.
    - [x] Add callbacks pushing state updates.
    - [x] Visualize workspace status in metrics dashboard.
    - [x] Test Global Workspace integration path.
315. [ ] Provide cross-device tensor synchronization to minimize latency during distributed training.
    - [ ] Design delta encoding format for tensors.
    - [ ] Implement diff-based synchronization protocol.
        - [ ] Compute tensor deltas using XOR for integer types and arithmetic
          difference for floating tensors.
        - [ ] Apply incoming deltas atomically on each device.
    - [ ] Add background sync service coordinating devices.
        - [ ] Spawn a lightweight worker thread per device to exchange deltas.
        - [ ] Aggregate metrics to detect and resync stale devices.
    - [ ] Tune synchronization interval via configuration.
        - [x] Add `sync.interval_ms` to configuration and CLI flags.
        - [x] Provide recommended ranges in YAML manual and parameters list.
    - [ ] Add tests measuring latency reduction.
        - [ ] Simulate a two-device setup with dummy tensors.
        - [ ] Verify synchronization reduces transfer volume and wall-clock time.
316. [x] Expose a low-level API to monitor event bus traffic for debugging.
    - [x] Add debug hooks to subscribe to raw events.
    - [x] Provide filtering and rate limiting options.
    - [x] Document usage in developer guide.
    - [x] Add tests ensuring hooks have minimal overhead.
317. [x] Review all documentation for completeness regarding new features such as DatasetCacheServer and remote hardware plugins.
    - [x] Audit existing docs for missing references.
    - [x] Add sections for DatasetCacheServer.
    - [x] Document remote hardware plugin API usage.
    - [x] Ensure tutorial and README mention new features.
318. [x] Update remaining markdown files to reference the remote hardware plugin API where relevant.
    - [x] Search repository for outdated plugin references.
    - [x] Update examples and configuration docs.
    - [x] Cross-link plugin API from relevant guides.
    - [x] Run lint checks to ensure markdown consistency.
319. [ ] Integrate JAX backend for differentiability.
    - [x] Create backend abstraction layer (`tensor_backend.py`) with functions like `matmul`, `sigmoid`, and `relu`.
    - [x] Implement NumPy and JAX backend versions.
    - [ ] Refactor Mandelbrot seed generation (`core/init_seed.py`) to use backend functions.
        - [ ] Replace direct NumPy operations with backend wrappers.
        - [ ] Ensure random seeds are produced via the selected backend.
        - [ ] Add unit tests confirming identical seeds across backends.
    - [ ] Update message passing (`core/message_passing.py`) to rely on the abstraction.
        - [ ] Swap low-level tensor ops for backend calls.
        - [ ] Enable runtime selection of NumPy or JAX paths.
        - [ ] Verify message delivery equivalence with tests.
    - [x] Add `core.backend` to `config.yaml` with fallback to NumPy and document in `yaml-manual.txt` and `CONFIGURABLE_PARAMETERS.md`.
    - [x] Test Mandelbrot output consistency across backends.

320. [ ] Introduce parallel Neuronenblitz workers.
    - [ ] Refactor `Neuronenblitz.train_example()` to be stateless and reentrant.
        - [ ] Remove global state dependencies.
        - [ ] Guard internal structures for thread safety.
        - [ ] Document reentrant assumptions in docstrings.
    - [ ] Implement `train_in_parallel()` using `concurrent.futures.ThreadPoolExecutor`.
        - [ ] Create worker wrapper around `train_example`.
        - [ ] Manage executor lifecycle and exception handling.
        - [ ] Aggregate gradients and metrics from workers.
    - [ ] Add `neuronenblitz.parallel_wanderers` to config with default 1 and document it.
        - [ ] Add parameter to `config.yaml` and CLI.
        - [ ] Describe usage in `yaml-manual.txt` and `CONFIGURABLE_PARAMETERS.md`.
        - [ ] Provide defaults and edge-case guidance.
    - [ ] Log worker-level metrics like average path length and divergence.
        - [ ] Record metrics within each worker loop.
        - [ ] Aggregate and report per-worker summaries.
    - [ ] Benchmark speedup on multi-core CPUs.
        - [ ] Design benchmark comparing single vs multi-worker.
        - [ ] Capture wall-clock time and throughput.
        - [ ] Summarize findings in docs.
    - [ ] Write tests for parallel wanderers.
        - [ ] Ensure deterministic results with one worker.
        - [ ] Validate scaling behavior with multiple workers.

321. [ ] Add quantization and sparse tensor support.
    - [x] Implement `QuantizedTensor` class with `.to_dense()` and `.to_bits()`.
        - [x] Define bit-packing strategy for tensors.
        - [x] Implement conversion methods for CPU and GPU.
        - [x] Provide serialization helpers.
    - [ ] Add optional quantization in `DataCompressor` using `core.quantization_bits` and document parameter.
        - [ ] Integrate `QuantizedTensor` into compression pipeline.
        - [ ] Add configuration option for bit width.
        - [ ] Update docs outlining trade-offs.
    - [ ] Use `scipy.sparse` for large synapse matrices.
        - [ ] Convert eligible matrices to sparse format.
        - [ ] Provide utilities to switch between dense and sparse.
        - [ ] Benchmark memory savings.
    - [ ] Validate lossless forward pass for common operations.
        - [ ] Compare quantized and dense outputs on sample layers.
        - [ ] Track numerical error and performance.
    - [ ] Add CLI flag `--quantize` for toggling quantization.
        - [ ] Parse flag and map to configuration value.
        - [ ] Document usage in CLI help and README.
    - [ ] Write unit tests verifying quantization correctness.
        - [ ] Test bit conversion round-trips.
        - [ ] Validate sparse and dense paths produce same results.

322. [ ] Implement causal attention and gating.
    - [ ] Add `core.attention_causal` to configuration and document it.
        - [ ] Insert parameter into `config.yaml` with default setting.
        - [ ] Extend `yaml-manual.txt` and `CONFIGURABLE_PARAMETERS.md`.
        - [ ] Provide CLI flag to toggle causal attention.
    - [ ] Modify attention mechanism to mask future tokens (`mask[i, j] = j > i`).
        - [ ] Implement mask generation routine.
        - [ ] Ensure masking works on CPU and GPU backends.
        - [ ] Benchmark overhead introduced by masking.
    - [ ] Implement `gating_layer` using sine or chaotic modulation.
        - [ ] Prototype gating function and parameter ranges.
        - [ ] Integrate gating into attention pipeline.
        - [ ] Expose gating parameters in configuration.
    - [ ] Visualize mask and gate effects on message propagation.
        - [ ] Create plotting utilities for masks and gates.
        - [ ] Add Streamlit view to display visuals.
    - [ ] Add tests for causal attention and gating behavior.
        - [ ] Verify masks prevent future token access.
        - [ ] Check gating outputs remain within expected bounds.

323. [x] Build streaming tokenizer and data loader.
    - [x] Refactor tokenizer interface to yield `tokenize(line)` instead of whole corpus.
    - [x] Implement `StreamingCSVLoader` in `dataset_loader.py`.
    - [x] Add resume token to track current byte offset and store metadata in `.meta.json`.
    - [x] Document streaming loader in tutorials and config manuals.
    - [x] Add tests for streaming tokenization and loader functionality.

324. [ ] Enhance Theory of Mind module.
    - [ ] Add `agent_id` and `belief_state` fields to input.
        - [ ] Extend input schemas and validation.
        - [ ] Update serialization/deserialization logic.
    - [ ] Encode beliefs as key-value memory slots in a new `ToMModule`.
        - [ ] Define memory slot structure and capacity.
        - [ ] Integrate module into existing pipeline.
    - [ ] Add multi-hop attention over belief states.
        - [ ] Implement attention layers supporting multiple hops.
        - [ ] Tune hop count for efficiency.
    - [ ] Log belief mismatches during evaluation.
        - [ ] Define mismatch metric and thresholds.
        - [ ] Store mismatches for post-run analysis.
    - [ ] Write tests for belief encoding and attention.
        - [ ] Unit test memory slot creation and retrieval.
        - [ ] Validate attention selects correct belief states.

325. [ ] Implement self-distillation over time.
    - [ ] Save `logits.pkl` after each epoch.
        - [ ] Hook training loop to capture logits.
        - [ ] Serialize logits with epoch metadata.
    - [ ] Add `self_distill_loss = KL(current_logits, previous_logits)` to the loss function with weight `meta_learning.distill_alpha`.
        - [ ] Implement KL divergence term.
        - [ ] Introduce `distill_alpha` parameter in config and docs.
    - [ ] Visualize alignment of predictions over time.
        - [ ] Plot KL divergence per epoch.
        - [ ] Display visualization in Streamlit dashboard.
    - [ ] Document distillation parameter in YAML manual and tutorial.
        - [ ] Describe purpose and recommended ranges.
        - [ ] Add tutorial section with example usage.
    - [ ] Add tests for self-distillation loss.
        - [ ] Unit test loss calculation with synthetic logits.
        - [ ] Ensure training uses previous epoch logits when available.

326. [ ] Create in-context learning prompt system.
    - [ ] Add `PromptMemory` to cache recent `(input, output)` pairs.
        - [x] Define `PromptMemory` class skeleton.
        - [x] Implement bounded queue with eviction policy.
        - [x] Add methods for retrieval and composite prompt generation.
        - [x] Provide serialization for stored pairs.
    - [ ] Modify inference to use `prompt + input` as composite query.
        - [x] Concatenate prompts with new inputs before inference.
        - [x] Handle empty or oversized prompt caches gracefully.
    - [ ] Add GUI control for toggling prompt injection.
        - [ ] Create Streamlit toggle linked to inference pipeline.
        - [ ] Persist user preference between sessions.
    - [ ] Store prompts persistently with timestamps.
        - [ ] Save prompt memory to disk on shutdown.
        - [x] Include timestamps for chronological retrieval.
    - [ ] Write tests for prompt cache behavior.
        - [x] Verify FIFO eviction policy.
        - [x] Test inference output when prompts are applied.

327. [ ] Add YAML config editor in Streamlit.
    - [ ] Add new "Config Editor" tab using `st_ace` with YAML syntax.
        - [x] Extract helper functions for loading and saving config files.
        - [x] Display `config.yaml` in `st_ace` with syntax highlighting and line numbers.
        - [x] Add save button invoking helper functions.
    - [ ] Validate YAML with schema on submit and save edits to `config.yaml` with backup timestamp.
        - [x] Validate YAML and create timestamped backup during save.
        - [x] Surface validation success or errors in UI.
    - [ ] Update GUI tests for the editor tab.
        - [x] Unit test configuration load/save helpers.
        - [ ] Simulate editing and saving a valid config.
        - [ ] Confirm invalid YAML triggers error messages in UI.

328. [ ] Integrate hyperparameter optimisation via Optuna.
    - [ ] Add `scripts/optimize.py` with Optuna study and objective function training one epoch.
        - [ ] Define search space and objective returning validation loss.
        - [ ] Allow resuming existing studies.
    - [ ] Log trials to `optuna_db.sqlite3`.
        - [ ] Configure SQLite storage backend.
        - [ ] Expose path via CLI option.
    - [ ] Add Streamlit tab to visualize optimization history and best config.
        - [ ] Plot trial scores and parameter importances.
        - [ ] Provide download of best config.
    - [ ] Document usage in tutorials and manuals.
        - [ ] Include setup and run instructions.
        - [ ] Describe interpretation of optimization charts.
    - [ ] Add tests for optimization script.
        - [ ] Use tiny dataset to run a short study.
        - [ ] Assert database and result files are created.

329. [ ] Implement live neuron graph visualisation.
    - [ ] Export graph as `{"nodes": [...], "edges": [...]}` in core.
        - [ ] Gather neuron and synapse metadata into structured dicts.
        - [ ] Support incremental updates for dynamic graphs.
    - [ ] Add `/graph` API endpoint.
        - [ ] Implement endpoint returning serialized graph JSON.
        - [ ] Secure endpoint with optional authentication.
    - [ ] Render graph with `plotly.graph_objects.Sankey` or `pyvis.network` and add sliders for filtering.
        - [ ] Build reusable visualization component.
        - [ ] Provide sliders for weight and degree thresholds.
    - [ ] Update GUI tests for graph visualization.
        - [ ] Confirm endpoint availability in tests.
        - [ ] Validate sliders adjust visible graph elements.

330. [ ] Introduce multi-agent MARBLE.
    - [ ] Define `MARBLEAgent` wrapper with its own config and brain.
        - [ ] Encapsulate brain initialization and lifecycle methods.
        - [ ] Allow per-agent configuration loading.
    - [ ] Implement `MessageBus` for agent-to-agent communication.
        - [ ] Design protocol for broadcasting and direct messages.
        - [ ] Ensure thread-safe message queues.
    - [ ] Simulate cooperative or competitive RL environments.
        - [ ] Provide sample environment harnesses.
        - [ ] Support reward sharing and competition modes.
    - [ ] Log inter-agent influence and conversation.
        - [ ] Record message histories and effect metrics.
        - [ ] Visualize interactions in dashboard.
    - [ ] Add tests for multi-agent interactions.
        - [ ] Unit test message exchange between two agents.
        - [ ] Validate environment simulations run without deadlocks.

331. [ ] Add evolutionary learning module.
    - [ ] Create `EvolutionTrainer` class.
        - [ ] Implement mutation, evaluation, and selection hooks.
        - [ ] Support parallel evaluation of candidates.
    - [ ] Generate `N` config mutations, train each for a few steps, and evaluate fitness.
        - [ ] Define mutation operators for numeric and categorical params.
        - [ ] Collect fitness metrics after partial training.
    - [ ] Select top `M` configurations and mutate again.
        - [ ] Rank configurations by fitness.
        - [ ] Produce next-generation configs via mutation.
    - [ ] Log evolution tree and best lineage.
        - [ ] Track parent-child relationships.
        - [ ] Serialize lineage to JSON and graphs.
    - [ ] Add tests for evolutionary training.
        - [ ] Unit test mutation and selection logic.
        - [ ] Integration test over multiple generations.

332. [ ] Document new configuration parameters and tutorials.
    - [ ] Update `yaml-manual.txt` with detailed explanations and examples.
        - [ ] Describe new parameters and their ranges.
        - [ ] Include practical usage examples.
    - [ ] Add entries to `CONFIGURABLE_PARAMETERS.md`.
        - [ ] List default values and descriptions.
        - [ ] Cross-reference related parameters.
    - [ ] Extend `TUTORIAL.md` with projects for new features (e.g., self-distillation).
        - [ ] Create step-by-step project showcasing feature.
        - [ ] Link to dataset download and preparation code.

333. [ ] Add unit tests and verify CUDA fallbacks.
    - [ ] Write tests for quantization correctness, parallel wanderers, and prompt cache behavior.
        - [x] Cover QuantizedTensor round-trip accuracy.
            - [x] Create quantized tensors with known values.
            - [x] Convert to float and back and assert equality.
        - [x] Ensure parallel wanderers yield consistent results.
            - [x] Set up small world simulation with multiple wanderers.
            - [x] Compare outputs and convergence metrics.
        - [x] Test prompt cache operations under load.
            - [x] Fill cache with synthetic prompts.
            - [x] Measure latency and eviction behaviour.
    - [ ] Verify CUDA fallbacks for all new modules.
        - [x] QuantizedTensor CPU path.
            - [x] Force CPU execution for quantized tensor tests.
            - [x] Confirm outputs match GPU path.
        - [x] Parallel wanderers CPU fallback.
            - [x] Disable CUDA and run wanderer tests.
            - [x] Document any discrepancies (none observed).
        - [ ] PromptMemory CPU performance baseline.
            - [ ] Run load tests with CUDA disabled.
            - [ ] Compare timings with GPU-enabled runs.
        - [ ] Document any GPU-only limitations.
            - [ ] Record modules lacking CPU implementation.
            - [ ] Update README with limitation notes.

### Dream Replay Enhancements

- [ ] Tag training and ingestion pathways with emotion, arousal and stress values.
- [ ] Expose configurable weighting functions for dream replay beyond linear and exponential.
- [ ] Implement mental housekeeping to prune low-importance connections during dreams.
- [ ] Add short-term instant replay buffer and merge into long-term buffer.
- [ ] Orchestrate dream scheduler combining replay, weighting and housekeeping steps.
- [ ] Persist replay buffers and neuromodulatory state in model snapshots.
- [ ] Create integration tests verifying dreaming state survives save/load cycles.
- [ ] Benchmark learning performance with and without dream consolidation.
