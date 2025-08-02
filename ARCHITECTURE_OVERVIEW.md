# MARBLE Architecture Overview

This document provides a high level description of how the main components of
MARBLE interact with one another and how data flows through the system.
It complements the detailed parameter descriptions in `yaml-manual.txt`.

## Core Components

### Core
The `Core` stores the neurons and synapses that make up the graph.
When initialised it seeds the neuron values from a Mandelbrot fractal
or from a symbolic `formula` if provided.  Each neuron carries a
`representation` vector which is updated through message passing.
The core also manages tiered memory (VRAM/RAM/Disk) and can expand by
creating additional neurons and synapses when neurogenesis is triggered.

### Neuronenblitz
`Neuronenblitz` implements the dynamic wandering algorithm that traverses
the graph.  It performs learning via weight updates and structural
plasticity.  It also produces neuromodulatory feedback by exposing the
last message passing change and plasticity thresholds.

### Brain
`Brain` orchestrates training.  It owns a `Core`, a `Neuronenblitz`, a
`DataLoader` and optional utilities such as the `MetricsVisualizer` and
`MetaParameterController`.  During `Brain.train` it iteratively calls
`Neuronenblitz.train_example` on the provided examples, manages
neurogenesis and consolidation and updates metrics.

### Remote Hardware Plugins
MARBLE can offload computation to specialized devices through a plugin system. Provide a module defining `get_remote_tier` and set `remote_hardware.tier_plugin` in `config.yaml` to enable it. The included `GrpcRemoteTier` communicates with a gRPC service.

### Plugin System
Beyond hardware acceleration, MARBLE exposes a lightweight plugin framework.
Modules listed under the top-level `plugins` key in `config.yaml` are imported
on start-up and may register new neuron or synapse types, background services or
other utilities via a `register(brain)` function. This enables experimentation
without modifying the core codebase.

### Auxiliary Services
Several helper modules support large scale experiments. ``DatasetCacheServer``
shares preprocessed datasets over HTTP so repeated runs avoid redundant
downloads. ``distributed_training.DistributedTrainer`` wraps PyTorch's process
group API to synchronise weights across multiple workers. A
``metrics_dashboard.MetricsDashboard`` instance renders live charts in a browser
and the minimalist ``memory_manager.MemoryManager`` tracks upcoming allocations
to avoid oversubscription. The ``experiment_tracker`` module logs metrics to
external services such as Weights & Biases via ``WandbTracker`` so long-running
experiments remain reproducible.

``system_metrics`` exposes lightweight functions to query CPU, RAM and GPU
utilisation, while ``usage_profiler.UsageProfiler`` records these values to CSV
for long-running jobs. For serving predictions ``web_api.InferenceServer`` spins
up a minimal Flask application that exposes the active brain over HTTP so other
systems can submit inference requests.

``dataset_loader`` provides high level helpers for downloading, sharding and
caching tabular datasets while tracking dependencies and memory usage. Datasets
can be versioned with ``dataset_versioning`` which writes diffs for each change
and later applies them to reproduce exact states. ``dataset_replication``
pushes dataset files to multiple HTTP endpoints so distributed jobs start from
identical inputs.

## Data Compression Pipeline
The `DataLoader` converts arbitrary Python objects or arrays into binary
representations using the `DataCompressor` and can optionally cache them via
`BitTensorDataset`:

1. **Serialize** – objects are pickled or converted to NumPy arrays.
2. **Binary Conversion** – the raw bytes are turned into a stream of bits
   via `DataCompressor.bytes_to_bits`.
3. **Compression** – `zlib.compress` is applied with a configurable
   compression level.
4. **Decoding** performs the exact inverse: decompress, convert bits back
   to bytes and unpickle.  Cached datasets are keyed by a shared vocabulary so
   multiple processes can reuse encodings.  Deterministic splitting and optional
   on-disk caching allow large corpora to be reused across runs.

Throughout training the compression ratio and cache hit rate are logged in
`MetricsVisualizer` which allows monitoring of the overhead introduced by
compression and the effectiveness of caching.

## Component Interaction
1. Input data from the outside world is fed through `DataLoader.encode`
   before entering the brain.  The resulting tensor is used by
   `Neuronenblitz` during dynamic wandering or training.
2. Outputs from the graph are passed back through
   `DataLoader.decode` to reconstruct the original Python objects.
3. Neuromodulatory signals collected in `NeuromodulatorySystem` influence
   plasticity and neurogenesis decisions inside `Brain`.
4. The `MetaParameterController` observes validation loss and adjusts the
   plasticity threshold of `Neuronenblitz` over time.

A simplified flow diagram:

```
Input -> DataLoader.encode -> Brain.train/dynamic_wander ->
Core/Neuronenblitz -> DataLoader.decode -> Output
```

## Future Extensions
Future work could extend the compression layer to support streaming
data sources, integrate JAX based automatic differentiation for
alternative training schemes and provide more sophisticated memory
management policies.
