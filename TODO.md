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
168. [ ] Cache intermediate results so iterative experiments run faster.
    - [x] Create file based cache storing step outputs keyed by index and function name.
    - [x] Add `clear_cache` method to remove cached files when needed.
    - [x] Unit tests demonstrating that repeated runs reuse cached results.
169. [ ] Support checkpointing and resuming pipelines with dataset version tracking.
    - [x] Track `dataset_version` within `HighLevelPipeline` instances.
    - [x] Implement `save_checkpoint` and `load_checkpoint` methods.
    - [x] Test saving and loading pipelines with version metadata.
170. [ ] Provide interactive step visualisation in the Streamlit GUI using dataset introspection.
    - [x] Add a "Step Visualisation" expander showing step parameters and dataset info.
    - [x] Unit tests ensuring the new expander appears in the Pipeline tab.
171. [ ] Offer a plugin system so users can register custom pipeline steps easily.
172. [ ] Manage dependencies between steps automatically to maintain correct order.
173. [ ] Allow branching paths in a pipeline to explore alternative experiment flows.
174. [ ] Send real-time progress events to the GUI during pipeline execution.
175. [ ] Recover gracefully from remote failures with retry logic.
176. [ ] Execute steps on multiple processes while sharing datasets through the core.
177. [ ] Add pre and post hooks for each step enabling custom behaviour.
178. [ ] Provide templates to quickly generate common workflows.
179. [ ] Automatically build training loops for Neuronenblitz when dataset steps are present.
180. [ ] Offer a specialised step that consumes the new streaming `BitTensorDataset`.
181. [ ] Persist step results to disk for quick re-runs.
182. [ ] Visualise pipelines as graphs using the marble graph builder.
183. [ ] Limit GPU memory usage per step through concurrency controls.
184. [ ] Debug steps interactively by inspecting their inputs and outputs.
185. [ ] Export trained models automatically as a final pipeline step.
186. [ ] Log pipeline events to the remote experiment tracker.
187. [ ] Validate configuration of each step using marble core schemas.
188. [ ] Group multiple operations into macro steps for convenience.
189. [ ] Roll back to earlier step outputs when experiments go wrong.
190. [ ] Integrate hyperparameter search that plugs directly into the pipeline engine.
191. [ ] Schedule individual steps on remote hardware tiers seamlessly.
192. [ ] Distribute dataset shards across parallel pipelines.
193. [ ] Estimate resource needs ahead of execution to inform the memory manager.
194. [ ] Save run profiles capturing the exact execution order.
195. [ ] Edit pipeline definitions interactively through the GUI.
196. [ ] Relay dataset events to pipeline notifications.
197. [ ] Update Neuronenblitz models automatically when datasets change.
198. [ ] Provide built-in cross-validation loops using deterministic dataset splits.
199. [ ] Serve models through the web API directly from a pipeline step.
200. [x] Benchmark pipeline steps using the core micro-benchmark tool.
201. [ ] Reorder steps dynamically based on dependency resolution.
202. [x] Broadcast pipeline progress to the Global Workspace plugin.
203. [x] Route step logs to the metrics visualiser for real-time viewing.
204. [x] Diff pipeline configurations to track changes between runs.
205. [x] Stream logs from each step into the GUI console.
206. [x] Pre-allocate resources via the memory pool before executing steps.
207. [x] Freeze and defrost steps without removing them from the pipeline.
208. [ ] Run pipeline sections in isolated processes for fault tolerance.
209. [ ] Connect with remote wanderers for asynchronous exploration phases.
210. [ ] Secure pipeline data flow by integrating dataset encryption routines.
211. [ ] Route memory allocations through the memory pool for every operation.
212. [x] Provide a CLI wrapper so pipelines can run without writing Python code.
213. [x] Detect GPU availability and adapt pipeline behaviour automatically.
214. [x] Persist vocabulary mappings for reuse across multiple runs.
215. [ ] Train directly from streamed dataset shards loaded via pipeline steps.
216. [ ] Integrate HighLevelPipeline with the forthcoming Neuronenblitz improvements.
217. [ ] Support streaming dataset shards during Neuronenblitz training to keep the model responsive.
218. [ ] Allow learning modules to be swapped in and out through a plugin interface.
219. [ ] Use Global Workspace events to guide dynamic attention gating.
220. [ ] Provide a reinforcement learning loop coordinated by pipeline scheduling.
221. [ ] Offload wandering to remote hardware using Marble Core utilities.
222. [x] Optimise memory usage by sharing dataset caches with the memory pool.
223. [ ] Accumulate gradients asynchronously in line with pipeline scheduling.
224. [ ] Inspect neural pathways interactively via the GUI.
225. [ ] Register custom loss modules through the plugin system.
226. [ ] Transfer knowledge between models using dataset serialisation features.
227. [ ] Refresh vocabulary encodings mid-training when datasets evolve.
228. [ ] Evaluate models remotely using the pipeline inference plugin.
229. [ ] Plan actions hierarchically using Global Workspace goals.
230. [ ] Adjust curricula automatically based on dataset history.
231. [ ] Update plasticity parameters from dataset augmentation events.
232. [ ] Allocate weights from the GPU memory pool for efficient updates.
233. [ ] Step through the pipeline debugger during training runs.
234. [ ] Wander asynchronously while prefetching dataset shards.
235. [ ] Include a self-supervised module that consumes augmented data.
236. [ ] Share memory buffers across nodes with the remote memory pool.
237. [ ] Load synapse types dynamically via pipeline plugins.
238. [ ] Checkpoint models in a universal format understood by the pipeline.
239. [ ] Inject gradient noise derived from dataset noise augmentation.
240. [ ] Prune routes when deduplication removes redundant data.
241. [ ] Explore dataset samples in the interactive browser during training.
242. [ ] Warm-start models from partial pipeline outputs.
243. [ ] Adapt learning rates using events from the memory manager.
244. [ ] Decrypt encrypted data on the fly during training.
245. [ ] Aggregate gradients remotely using distributed helpers.
246. [ ] Retrieve activations using approximate nearest neighbour search.
247. [ ] Gating mechanisms use dataset tags for context.
248. [ ] Reinitialize parts of the network when datasets are patched.
249. [ ] Evaluate intermediate models using generated validation sets.
250. [ ] Add runtime extension for registering new neuron types.
251. [ ] Communicate between models using Global Workspace broadcast.
252. [ ] Enable dropout gating informed by dataset quality hooks.
253. [ ] Derive reinforcement signals from pipeline event logs.
254. [ ] Mix offline and online learning phases based on dataset modifications.
255. [ ] Integrate with hyperparameter sweeps to evaluate multiple runs.
256. [ ] Use the memory pool for gradient buffer management.
257. [ ] Route decisions based on hierarchical dataset tags.
258. [ ] Cache activations for repeated passes over dataset shards.
259. [x] Send training events to the metrics visualiser.
    - [x] Add log_event method to MetricsVisualizer.
    - [x] Emit epoch_start and epoch_end in Brain.train.
    - [x] Display events in GUI console.
260. [ ] Ensure encrypted datasets remain private throughout training.
261. [ ] Replicate networks across nodes using remote hardware plugins.
262. [ ] Quickly evaluate models via pipeline checkpoint restore.
263. [ ] Roll back neuron states when dataset audits fail.
264. [ ] Adapt gating when vocabulary updates occur.
265. [ ] Manage experiments across runs using pipeline and dataset versions.
266. [ ] Build a distributed memory pool so datasets are shared across nodes.
267. [ ] Accelerate bit tensor operations on GPU for faster dataset processing.
268. [ ] Provide a remote procedure interface to run pipeline steps asynchronously.
269. [ ] Implement a system-wide event bus linking dataset, pipeline and Neuronenblitz modules.
270. [ ] Offer encryption utilities supporting secure dataset storage.
271. [ ] Handle memory-mapped tensors for large streaming datasets.
272. [ ] Load configuration hierarchies merging dataset and pipeline settings.
273. [ ] Add backpressure-aware message passing for streaming data.
274. [ ] Create a cross-process checkpoint manager accessible from pipelines.
275. [ ] Dynamically load neuron and synapse plugins at runtime.
276. [ ] Debug message passing flows in real time through GUI tools.
277. [ ] Provide a remote scheduler for distributed pipeline execution.
278. [ ] Verify data integrity across modules with unified checksum tools.
279. [ ] Serialise models and datasets through a cross-platform layer.
280. [ ] Cache activation tensors that Neuronenblitz reuses during training.
281. [ ] Securely store encryption keys via a dedicated security manager.
282. [ ] Hot‑reload modules without stopping ongoing training.
283. [ ] Manage GPU memory for dataset prefetching.
284. [ ] Support asynchronous I/O for continuous dataset streaming.
285. [ ] Emit resource events to notify components of memory pressure.
286. [ ] Replicate memory pools across machines for distributed workloads.
287. [x] Validate configurations globally before starting a pipeline.
    - [x] Implement validate_global_config function.
    - [x] Add schema checks for all sections.
288. [ ] Provide thread-safe APIs for parallel dataset transformations.
289. [ ] Deduplicate repeated bit tensors across different datasets.
290. [ ] Aggregate event logs from all modules into a unified stream.
291. [ ] Update models with partial graph recompilation.
292. [ ] Accelerate nearest neighbour search via specialised hardware plugins.
293. [ ] Queue wander jobs reliably even if a process crashes.
294. [ ] Synchronise checkpoints automatically across nodes.
295. [ ] Schedule steps using a GPU-aware dispatcher.
296. [ ] Prune unused neurons according to dataset prune events.
297. [ ] Propagate context from pipeline events into core behaviours.
298. [ ] Visualise core operations in a built-in GUI server.
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
    - [ ] Expose pipeline metrics to Global Workspace.
    - [ ] Add callbacks pushing state updates.
    - [ ] Visualize workspace status in metrics dashboard.
    - [ ] Test Global Workspace integration path.
315. [ ] Provide cross-device tensor synchronization to minimize latency during distributed training.
    - [ ] Implement diff-based synchronization protocol.
    - [ ] Add background sync service coordinating devices.
    - [ ] Tune synchronization interval via configuration.
    - [ ] Add tests measuring latency reduction.
316. [ ] Expose a low-level API to monitor event bus traffic for debugging.
    - [ ] Add debug hooks to subscribe to raw events.
    - [ ] Provide filtering and rate limiting options.
    - [ ] Document usage in developer guide.
    - [ ] Add tests ensuring hooks have minimal overhead.
317. [ ] Review all documentation for completeness regarding new features such as DatasetCacheServer and remote hardware plugins.
    - [ ] Audit existing docs for missing references.
    - [ ] Add sections for DatasetCacheServer.
    - [ ] Document remote hardware plugin API usage.
    - [ ] Ensure tutorial and README mention new features.
318. [ ] Update remaining markdown files to reference the remote hardware plugin API where relevant.
    - [ ] Search repository for outdated plugin references.
    - [ ] Update examples and configuration docs.
    - [ ] Cross-link plugin API from relevant guides.
    - [ ] Run lint checks to ensure markdown consistency.
319. [ ] Integrate JAX backend for differentiability.
    - [ ] Create backend abstraction layer (`tensor_backend.py`) with functions like `matmul`, `sigmoid`, and `relu`.
    - [ ] Implement NumPy and JAX backend versions.
    - [ ] Refactor Mandelbrot seed generation (`core/init_seed.py`) to use backend functions.
    - [ ] Update message passing (`core/message_passing.py`) to rely on the abstraction.
    - [ ] Add `core.backend` to `config.yaml` with fallback to NumPy and document in `yaml-manual.txt` and `CONFIGURABLE_PARAMETERS.md`.
    - [ ] Test Mandelbrot output consistency across backends.

320. [ ] Introduce parallel Neuronenblitz workers.
    - [ ] Refactor `Neuronenblitz.train_example()` to be stateless and reentrant.
    - [ ] Implement `train_in_parallel()` using `concurrent.futures.ThreadPoolExecutor`.
    - [ ] Add `neuronenblitz.parallel_wanderers` to config with default 1 and document it.
    - [ ] Log worker-level metrics like average path length and divergence.
    - [ ] Benchmark speedup on multi-core CPUs.
    - [ ] Write tests for parallel wanderers.

321. [ ] Add quantization and sparse tensor support.
    - [ ] Implement `QuantizedTensor` class with `.to_dense()` and `.to_bits()`.
    - [ ] Add optional quantization in `DataCompressor` using `core.quantization_bits` and document parameter.
    - [ ] Use `scipy.sparse` for large synapse matrices.
    - [ ] Validate lossless forward pass for common operations.
    - [ ] Add CLI flag `--quantize` for toggling quantization.
    - [ ] Write unit tests verifying quantization correctness.

322. [ ] Implement causal attention and gating.
    - [ ] Add `core.attention_causal` to configuration and document it.
    - [ ] Modify attention mechanism to mask future tokens (`mask[i, j] = j > i`).
    - [ ] Implement `gating_layer` using sine or chaotic modulation.
    - [ ] Visualize mask and gate effects on message propagation.
    - [ ] Add tests for causal attention and gating behavior.

323. [ ] Build streaming tokenizer and data loader.
    - [ ] Refactor tokenizer interface to yield `tokenize(line)` instead of whole corpus.
    - [ ] Implement `StreamingCSVLoader` in `dataset_loader.py`.
    - [ ] Add resume token to track current byte offset and store metadata in `.meta.json`.
    - [ ] Document streaming loader in tutorials and config manuals.
    - [ ] Add tests for streaming tokenization and loader functionality.

324. [ ] Enhance Theory of Mind module.
    - [ ] Add `agent_id` and `belief_state` fields to input.
    - [ ] Encode beliefs as key-value memory slots in a new `ToMModule`.
    - [ ] Add multi-hop attention over belief states.
    - [ ] Log belief mismatches during evaluation.
    - [ ] Write tests for belief encoding and attention.

325. [ ] Implement self-distillation over time.
    - [ ] Save `logits.pkl` after each epoch.
    - [ ] Add `self_distill_loss = KL(current_logits, previous_logits)` to the loss function with weight `meta_learning.distill_alpha`.
    - [ ] Visualize alignment of predictions over time.
    - [ ] Document distillation parameter in YAML manual and tutorial.
    - [ ] Add tests for self-distillation loss.

326. [ ] Create in-context learning prompt system.
    - [ ] Add `PromptMemory` to cache recent `(input, output)` pairs.
    - [ ] Modify inference to use `prompt + input` as composite query.
    - [ ] Add GUI control for toggling prompt injection.
    - [ ] Store prompts persistently with timestamps.
    - [ ] Write tests for prompt cache behavior.

327. [ ] Add YAML config editor in Streamlit.
    - [ ] Add new "Config Editor" tab using `st_ace` with YAML syntax.
    - [ ] Validate YAML with schema on submit and save edits to `config.yaml` with backup timestamp.
    - [ ] Update GUI tests for the editor tab.

328. [ ] Integrate hyperparameter optimisation via Optuna.
    - [ ] Add `scripts/optimize.py` with Optuna study and objective function training one epoch.
    - [ ] Log trials to `optuna_db.sqlite3`.
    - [ ] Add Streamlit tab to visualize optimization history and best config.
    - [ ] Document usage in tutorials and manuals.
    - [ ] Add tests for optimization script.

329. [ ] Implement live neuron graph visualisation.
    - [ ] Export graph as `{"nodes": [...], "edges": [...]}` in core.
    - [ ] Add `/graph` API endpoint.
    - [ ] Render graph with `plotly.graph_objects.Sankey` or `pyvis.network` and add sliders for filtering.
    - [ ] Update GUI tests for graph visualization.

330. [ ] Introduce multi-agent MARBLE.
    - [ ] Define `MARBLEAgent` wrapper with its own config and brain.
    - [ ] Implement `MessageBus` for agent-to-agent communication.
    - [ ] Simulate cooperative or competitive RL environments.
    - [ ] Log inter-agent influence and conversation.
    - [ ] Add tests for multi-agent interactions.

331. [ ] Add evolutionary learning module.
    - [ ] Create `EvolutionTrainer` class.
    - [ ] Generate `N` config mutations, train each for a few steps, and evaluate fitness.
    - [ ] Select top `M` configurations and mutate again.
    - [ ] Log evolution tree and best lineage.
    - [ ] Add tests for evolutionary training.

332. [ ] Document new configuration parameters and tutorials.
    - [ ] Update `yaml-manual.txt` with detailed explanations and examples.
    - [ ] Add entries to `CONFIGURABLE_PARAMETERS.md`.
    - [ ] Extend `TUTORIAL.md` with projects for new features (e.g., self-distillation).

333. [ ] Add unit tests and verify CUDA fallbacks.
    - [ ] Write tests for quantization correctness, parallel wanderers, and prompt cache behavior.
    - [ ] Verify CUDA fallbacks for all new modules.
