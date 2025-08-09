# CPU Fallback Strategy and Catalog

## Fallback strategy
Marble provides GPU acceleration when available but every module must
offer an equivalent CPU execution path. Each component guards CUDA
usage with `torch.cuda.is_available()` and routes tensors accordingly
so behaviour remains consistent regardless of device. Parity tests
assert that CPU and GPU paths produce matching results, and the CI
workflow `.github/workflows/ci.yml` runs on CPU-only runners to detect
regressions in fallback logic.

## Verified CPU/GPU parity
The following modules currently have explicit tests confirming parity
between devices:
- `attention_utils.py` – see `tests/test_attention_utils_cpu_fallback.py`,
- `benchmark_graph_precompile.py` – see `tests/test_benchmark_graph_precompile_cpu_gpu.py`,
- `run_profiler.py` – see `tests/test_run_profiler_cpu_gpu.py`,
- `soft_actor_critic.py` – see `tests/test_soft_actor_critic.py`.

## Modules requiring CPU fallback tests
Generated list of modules referencing CUDA-specific code paths.

- adversarial_dataset.py
- adversarial_generator.py
- adversarial_learning.py
- async_gradient_accumulator.py
- async_utils.py
- attention_codelets.py
- attention_utils.py
- benchmark_config_training.py
- benchmark_graph_precompile.py
- bit_tensor_dataset.py
- branch_container.py
- config_loader.py
- convert_model.py
- cross_validation.py
- evolution_trainer.py
- exampletrain.py
- highlevel_pipeline.py
- highlevel_pipeline_cli.py
- marble.py
- marble_activation_kernel.py
- marble_brain.py
- marble_imports.py
- marble_main.py
- marble_neuronenblitz/attention_span.py
- marble_neuronenblitz/core.py
- marble_utils.py
- mcp_server.py
- memory_pool.py
- model_refresh.py
- neural_pathway.py
- neuronenblitz_kernel.py
- pipeline.py
- process_manager.py
- reinforcement_learning.py
- run_profiler.py
- scripts/optimize.py
- streaming_dataset_step.py
- streamlit_playground.py
- system_metrics.py
- tests/branching_steps.py
- tests/test_async_gradient_accumulator.py
- tests/test_attention_span_module.py
- tests/test_attention_workspace_gating.py
- tests/test_auto_nb_training_loop.py
- tests/test_autograd_layer.py
- tests/test_branch_container_concurrency.py
- tests/test_config_training_benchmark.py
- tests/test_context_attention.py
- tests/test_cross_validation.py
- tests/test_custom_loss_validation.py
- tests/test_dataset_transfer.py
- tests/test_experiment_tracker_event_logging.py
- tests/test_gpu_parity.py
- tests/test_graph_module_conversion.py
- tests/test_highlevel_pipeline.py
- tests/test_highlevel_pipeline_async_benchmark.py
- tests/test_highlevel_pipeline_cache.py
- tests/test_highlevel_pipeline_cli.py
- tests/test_highlevel_pipeline_features.py
- tests/test_highlevel_pipeline_neuronenblitz.py
- tests/test_hyperparameter_search.py
- tests/test_interactive_debug.py
- tests/test_kuzu_experiment_tracker.py
- tests/test_marble_activation_kernel.py
- tests/test_mcp_serve_model_plugin.py
- tests/test_mixed_precision.py
- tests/test_multiprocessing_dataset.py
- tests/test_nb_streaming_shards.py
- tests/test_neural_pathway_inspector.py
- tests/test_neuronenblitz_auto_update.py
- tests/test_neuronenblitz_cuda.py
- tests/test_ollama_pipeline_plugin.py
- tests/test_parallel_dataset_shards.py
- tests/test_pipeline_cache.py
- tests/test_pipeline_cache_stress.py
- tests/test_pipeline_dependency_integration.py
- tests/test_pipeline_hooks.py
- tests/test_pipeline_macro_rollback.py
- tests/test_pipeline_resume_integration.py
- tests/test_prompt_memory_cpu.py
- tests/test_pytorch_to_marble.py
- tests/test_quantized_tensor.py
- tests/test_remote_step_scheduling.py
- tests/test_resource_estimation.py
- tests/test_rnn_hidden_state_serialization.py
- tests/test_run_profile.py
- tests/test_serve_model_plugin.py
- tests/test_shared_vocab.py
- tests/test_streamed_training.py
- tests/test_streaming_dataset_step.py
- tests/test_tensor_sync_service.py
- tests/test_transparent_layer.py
- tests/test_workflow_template_generator.py
- unified_learning.py
- workflow_template_generator.py
