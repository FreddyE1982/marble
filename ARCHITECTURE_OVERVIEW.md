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

## Data Compression Pipeline
The `DataLoader` converts arbitrary Python objects or arrays into binary
tensors using the `DataCompressor`:

1. **Serialize** – objects are pickled or converted to NumPy arrays.
2. **Binary Conversion** – the raw bytes are turned into a stream of bits
   via `DataCompressor.bytes_to_bits`.
3. **Compression** – `zlib.compress` is applied with a configurable
   compression level.
4. **Decoding** performs the exact inverse: decompress, convert bits back
   to bytes and unpickle.  This guarantees full transitivity so that the
   original object is recovered exactly.

Throughout training the compression ratio is logged in
`MetricsVisualizer` which allows monitoring of the overhead introduced by
compression.

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
