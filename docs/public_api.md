# Public API Overview

This document lists the main classes and functions intended for external use.

- `marble_core.Core`
- `marble_core.DataLoader`
- `marble_brain.Brain`
- `marble_main.MARBLE`
- `marble_neuronenblitz.Neuronenblitz`
- `marble_neuronenblitz.learning.enable_rl`
- `marble_neuronenblitz.learning.disable_rl`
- `marble_neuronenblitz.learning.rl_select_action`
- `marble_neuronenblitz.learning.rl_update`
- `marble_neuronenblitz.memory.decay_memory_gates`
- `global_workspace.activate`
- `global_workspace.GlobalWorkspace.publish`
- `global_workspace.GlobalWorkspace.subscribe`
- `attention_codelets.register_codelet`
- `attention_codelets.run_cycle`
- `theory_of_mind.activate`
- `predictive_coding.activate`
- `bit_tensor_dataset.BitTensorDataset`
- `fractal_dimension_learning.FractalDimensionLearner`
- `quantum_flux_learning.QuantumFluxLearner`
- `continuous_weight_field_learning.ContinuousWeightFieldLearner`
- `dream_reinforcement_learning.DreamReinforcementLearner`
- `neural_schema_induction.NeuralSchemaInductionLearner`
- `conceptual_integration.ConceptualIntegrationLearner`
- `distributed_training.DistributedTrainer`
- `dataset_cache_server.DatasetCacheServer`
- `remote_offload.RemoteBrainServer`
- `remote_offload.RemoteBrainClient`
- `event_bus.global_event_bus`
- `remote_hardware.base.RemoteTier`
- `remote_hardware.plugin_loader.load_plugin`
- `metrics_dashboard.MetricsDashboard`
- `memory_manager.MemoryManager`
- `config_sync_service.ConfigSyncService`
- `highlevel_pipeline.HighLevelPipeline`
- `dataset_loader.load_dataset`
- `dataset_loader.prefetch_dataset`
- `dataset_loader.export_dataset`
- `dataset_loader.load_kuzu_graph`
- `dataset_loader.load_training_data_from_config`
- `dataset_versioning.create_version`
- `dataset_versioning.apply_version`
- `dataset_replication.replicate_dataset`
- `graph_streaming.stream_graph_chunks`
- `pipeline.Pipeline`
- `model_quantization.quantize_core_weights`
- `experiment_tracker.ExperimentTracker`
- `experiment_tracker.WandbTracker`
- `system_metrics.profile_resource_usage`
- `usage_profiler.UsageProfiler`
- `web_api.InferenceServer`

These APIs are kept stable across minor versions. Internal helpers not listed here may change without notice.

## Event bus debugging

`event_bus.global_event_bus` exposes low-level access to MARBLE's internal
event stream. Developers can subscribe with callbacks that accept an event name
and payload dictionary:

```python
from event_bus import global_event_bus

def hook(name, data):
    print(name, data)

global_event_bus.subscribe(hook, events=["dataset_load_start"], rate_limit_hz=1)
```

The optional `events` argument filters by name while `rate_limit_hz` limits the
number of callbacks per second to minimise overhead.

## Remote hardware plugins

Remote tiers allow offloading heavy computations to external hardware. Create a
module exposing ``get_remote_tier(cfg)`` and set its import path in
``remote_hardware.tier_plugin``. The helper ``remote_hardware.plugin_loader.load_plugin``
returns an instance implementing the ``RemoteTier`` interface with
``run_lobe`` and ``close`` methods.
