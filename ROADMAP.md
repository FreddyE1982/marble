# Marble Roadmap

This document outlines upcoming milestones for future Marble releases.

## v0.4 – Q4 2024 *(released)*
- TensorFlow interoperability layer.
- Support for experiment collections in the configuration.
- Gradient accumulation for autograd training.
- Dataset cache server for sharing downloads across nodes.

## v0.5 – Q1 2025 *(released)*
- Distributed multi-GPU training utilities via ``DistributedTrainer``.
- Expanded template library for neurons and synapses.
- Docker images for streamlined deployment.

## v0.6 – Q3 2025 *(released)*
- Remote offloading with compression.
- Live metrics dashboard and memory manager.
- Additional BitTensor pipelines for imitation and fractal learning.

## v0.7 – Q1 2026
- Dataset versioning and replication utilities.
- Pipeline CLI for executing YAML-defined workflows.
- Initial model quantization helpers.

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
