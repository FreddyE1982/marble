# Roadmap for Marble, Marble Core, and Neuronenblitz Improvements

This TODO list outlines 100 enhancements spanning the Marble framework, the underlying Marble Core, and the Neuronenblitz learning system. The items are grouped by broad themes but are intentionally numbered for easy tracking.

1. [x] Expand unit test coverage across all modules.
2. [x] Implement continuous integration to automatically run tests on pushes.
3. [x] Improve error handling in `marble_core` for invalid neuron parameters.
4. [x] Add type hints to all functions for better static analysis.
   - [x] Add hints to marble_core.py
   - [x] Add hints to marble_neuronenblitz.py
   - [x] Add hints to streamlit_playground.py
5. Integrate GPU acceleration into all neural computations.
   - [x] Evaluate current modules for GPU compatibility.
   - [x] Implement GPU kernels using PyTorch operations and custom CUDA if needed.
   - [x] Provide CPU fallback mechanisms.
   - [x] Add tests verifying GPU and CPU parity.
6. [x] Provide a command line interface for common training tasks.
7. Refactor `marble_neuronenblitz.py` into logical submodules.
   - [x] Identify separate functionalities (learning algorithms, memory, etc.).
   - [x] Split file into modules under new package `marble_neuronenblitz/`.
   - [x] Update imports across the project.
   - [x] Add tests ensuring modules operate as before.
8. Document all public APIs with docstrings and examples.
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
26. Implement distributed training across multiple GPUs.
   - [ ] Research distributed training approaches (DDP, Horovod).
   - [ ] Add distributed setup utilities to `marble_core`.
   - [ ] Implement distributed training pipeline in `marble_neuronenblitz`.
   - [ ] Write tests using CPU-based simulator.
27. Provide higher-level wrappers for common reinforcement learning tasks.
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
49. Implement a plugin-based remote tier for custom hardware.
   - [x] Define plugin API for remote hardware tiers.
   - [x] Implement sample plugin using gRPC to remote server.
   - [x] Update configuration to select remote tiers.
   - [x] Add tests for plugin initialization and data transfer.
50. [x] Create visualization utilities for neuron activation patterns.
51. [x] Add parameter scheduling for exploration/exploitation trade-offs.
52. Support hierarchical reinforcement learning in Neuronenblitz.
   - [ ] Research HRL algorithms applicable to the architecture.
   - [x] Implement high-level action controller.
   - [x] Add low-level policy modules.
   - [x] Provide example training script.
53. Implement efficient memory management for huge graphs.
   - [ ] Identify memory-heavy structures.
   - [ ] Implement streaming / chunking of graph data.
   - [x] Add memory pooling and reference counting.
   - [ ] Benchmark and optimize.
54. [x] Add checks for NaN/Inf propagation throughout the core.
55. [x] Provide an option to profile CPU and GPU usage during training.
56. [x] Integrate dataset sharding for distributed training.
57. [x] Create a cross-platform installer script.
58. [x] Provide a simple web API for remote inference.
59. [x] Add command line tools to export trained models.
60. Implement automatic synchronization of config files across nodes.
   - [x] Design synchronization protocol (e.g., using rsync or file watchers).
   - [x] Implement config sync service.
   - [x] Add CLI command to trigger manual sync.
   - [x] Write tests simulating multi-node environment.
61. Enhance constant-time operations for cryptographic safety.
   - [x] Profile existing cryptographic operations.
   - [x] Replace variable-time functions with constant-time equivalents.
   - [x] Add unit tests verifying timing does not leak secrets.
   - [x] Document cryptographic safety guidelines.
62. Add more comprehensive adversarial training examples.
   - [ ] Implement adversarial example generators.
   - [ ] Add training loops demonstrating adversarial robustness.
   - [x] Provide dataset wrappers for adversarial data.
   - [ ] Document new examples in TUTORIAL.
63. [x] Provide utilities for automatic dataset downloading and caching.
64. [x] Integrate a simple hyperparameter search framework.
65. [x] Add tests verifying deterministic behaviour with fixed seeds.
66. Improve readability of configuration files with comments and sections.
   - [x] Group related config parameters into sections.
   - [x] Add descriptive comments for each parameter.
   - [x] Provide script to auto-generate sample config with comments.
   - [x] Update YAML manual accordingly.
67. [x] Implement graph pruning utilities to remove unused neurons.
68. [x] Create a repository of reusable neuron/synapse templates.
69. [x] Add support for mixed precision training when GPUs are available.
70. [x] Provide dynamic graph visualisation within the GUI.
71. [x] Implement scheduled backups of experiment logs and results.
72. Add a compatibility layer for older Python versions where feasible.
   - [x] Identify features incompatible with Python 3.8 and 3.9.
   - [x] Implement polyfills or wrappers.
   - [x] Setup CI matrix to test older versions.
   - [x] Document limitations.
73. [x] Provide self-contained Docker images for reproducibility.
74. [x] Implement offline mode with pre-packaged datasets.
75. Add automated packaging to publish releases on PyPI.
   - [x] Create setup.py and pyproject.toml for packaging.
   - [x] Setup CI workflow for building and uploading to TestPyPI.
   - [x] Add versioning scheme.
   - [x] Document release process.
76. [x] Improve data compression for network transfers.
77. Incorporate gradient accumulation for large batch training.
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
89. Implement context-aware attention mechanisms.
   - [ ] Research existing attention mechanisms.
   - [ ] Design architecture for context-aware attention.
   - [ ] Implement module in Neuronenblitz.
   - [ ] Provide unit tests and example usage.
90. [x] Add unit tests ensuring backward compatibility between versions.
91. [x] Create a `core_benchmark.py` script for micro benchmarks.
92. [x] Provide a template repository for new Marble-based projects.
93. [x] Add interactive tutorials in Jupyter notebooks.
94. [x] Expand the remote offload module with bandwidth estimation.
95. [x] Implement dynamic route optimisation in Neuronenblitz.
96. Add anomaly detection for wandering behaviour.
   - [x] Define metrics to measure wandering.
   - [x] Implement anomaly detection algorithm.
   - [x] Integrate with training logs.
   - [x] Add tests for detection accuracy.
97. [x] Provide visual feedback for training progress in Streamlit.
98. [x] [x] Offer integration examples with existing ML libraries.
99. [x] Enhance documentation with troubleshooting guides.
100. [x] Establish a long-term roadmap with release milestones.

101. Implement a **Global Workspace** plugin to broadcast conscious contents across all modules.
   - [x] Define data structures for global workspace broadcast.
   - [x] Implement plugin with message queue.
   - [x] Expose configuration options.
   - [x] Add tests verifying broadcast across modules.
102. Add **attention codelet** plugins that form coalitions and submit them to the Global Workspace.
   - [x] Create plugin interface for attention codelets.
   - [x] Implement coalition formation logic.
   - [x] Connect with Global Workspace plugin.
   - [x] Add example configuration.
103. Create a **Self-Monitoring** plugin that maintains an internal state model and meta-cognitive evaluations.
   - [ ] Design internal state data structures.
   - [ ] Implement self-monitoring algorithms.
   - [ ] Connect output to context_history.
   - [ ] Write tests for self-monitoring updates.
104. Integrate higher-order thought markers from Self-Monitoring into `context_history`.
   - [ ] Extend context_history data structure.
   - [ ] Add functions to log HOT markers.
   - [ ] Update Self-Monitoring plugin to emit markers.
   - [ ] Add tests verifying markers saved.
105. Link Self-Monitoring feedback to reinforcement learning and `dynamic_wander` adjustments.
   - [ ] Create interface between self-monitoring and RL modules.
   - [ ] Adjust dynamic_wander parameters based on self-monitoring output.
   - [ ] Provide configuration hooks.
   - [ ] Add integration tests.
106. Implement an **Episodic Memory** plugin supporting transient buffers, long‑term storage and context-based retrieval.
   - [ ] Design schemas for episodic entries.
   - [ ] Implement transient buffer and long-term storage.
   - [ ] Add retrieval API with context queries.
   - [ ] Write tests for storing and retrieving episodes.
107. Provide an **episodic simulation** plugin to replay stored episodes for planning.
   - [ ] Implement episode selection and playback engine.
   - [ ] Integrate with planning modules.
   - [ ] Provide configuration options for simulation length.
   - [ ] Document usage.
108. Develop a **Goal Manager** plugin handling hierarchical goals and conflict resolution with active inference.
   - [ ] Define goal hierarchy structures.
   - [ ] Implement conflict resolution algorithms.
   - [ ] Connect with reinforcement learning modules.
   - [ ] Add tests for typical goal scenarios.
109. Build a **Theory of Mind** plugin using character, mental-state and prediction subnets.
   - [ ] Research ToM models suitable for integration.
   - [ ] Implement subnets for character modelling and prediction.
   - [ ] Connect ToM to Global Workspace and Self-Monitoring.
   - [ ] Add example training script.
110. Implement a **Predictive Coding** plugin offering hierarchical predictions and active inference loops.
   - [ ] Design predictive coding architecture.
   - [ ] Implement hierarchical prediction modules.
   - [ ] Integrate with reinforcement learning and episodic memory.
   - [ ] Provide tests verifying prediction accuracy.
111. Expand `context_history` and `replay_buffer` to store internal markers, goals and ToM information.
   - [ ] Extend data structures to include markers, goals, ToM.
   - [ ] Update save/load logic.
   - [ ] Add migration for old checkpoints.
   - [ ] Write tests for new buffer behavior.
112. Extend attention mechanisms to interface with the Global Workspace and plugin salience scores.
   - [ ] Modify attention modules to accept salience inputs.
   - [ ] Connect to Global Workspace for broadcast.
   - [ ] Provide weight tuning parameters.
   - [ ] Add tests for attention with salience.
113. [x] Add YAML configuration options for all new plugins and document them thoroughly.
114. [x] Create unit and integration tests ensuring each plugin works on CPU and GPU.
115. Update tutorials and manuals with instructions on using the consciousness plugins.
   - [x] Write step-by-step tutorial for each new plugin.
   - [x] Update yaml-manual with plugin parameters.
   - [x] Add troubleshooting section.
   - [x] Provide example configuration files.

