# Marble Roadmap

This document outlines upcoming milestones for future Marble releases.

## v0.1.0 – Current Release
- TensorFlow interoperability layer
- Experiment collections in the configuration
- Gradient accumulation for autograd training
- Dataset cache server for sharing downloads across nodes
- Distributed multi-GPU training utilities via ``DistributedTrainer``
- Expanded neuron and synapse template library
- Docker images for streamlined deployment
- Remote offloading with compression
- Live metrics dashboard and memory manager
- BitTensor pipelines for imitation and fractal learning
- Dataset versioning and replication utilities
- Pipeline CLI for executing YAML-defined workflows
- Model quantization helpers
- System metrics module and usage profiler
- HTTP inference server for remote predictions
- Enhanced documentation sync across modules

## v0.2.0 – Planned
- Streamlined GUI integration across all features
  - Audit existing GUI components and identify duplication.
  - Ensure every feature toggles correctly on CPU and GPU.
  - Add automated tests covering each tab and widget.
- Additional tutorials and educational projects
  - Curate real-world datasets with download scripts.
  - Write step-by-step guides with full code listings.
  - Provide expected results for CPU and GPU runs.
- Extended hardware plugin catalog
  - Research upcoming accelerators (e.g., TPU, custom ASICs).
  - Prototype plugin interface for each device with CPU fallback.
  - Document installation and testing procedures.

Further versions will refine the API and add more tutorials based on
community feedback.

## Long Term Vision

The following milestones provide a high level view of Marble's evolution over
the coming years. Dates are tentative and may shift based on community needs.

- **v1.0 – Q3 2025**
  - Stable API with semantic versioning.
  - Official template repository published on PyPI.
  - Complete GUI test coverage across all features.
- **v1.5 – Q1 2026**
  - Initial support for distributed training across clusters.
  - Offline mode enhancements with dataset packaging utilities.
- **v2.0 – Q4 2026**
  - Context-aware attention mechanisms.
  - Full reinforcement learning suite including hierarchical methods.
  - Remote hardware plugin system supporting gRPC accelerators.
