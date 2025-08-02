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
- `distributed_training.DistributedTrainer`
- `dataset_cache_server.DatasetCacheServer`
- `remote_offload.RemoteBrainServer`
- `remote_offload.RemoteBrainClient`
- `metrics_dashboard.MetricsDashboard`
- `memory_manager.MemoryManager`
- `config_sync_service.ConfigSyncService`
- `highlevel_pipeline.HighLevelPipeline`
- `dataset_loader.load_dataset`
- `dataset_loader.prefetch_dataset`
- `dataset_loader.export_dataset`
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
