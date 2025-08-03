# Distributed Training Approaches

The Marble framework supports running Neuronenblitz models across multiple
processes using PyTorch's DistributedDataParallel (DDP). DDP provides efficient
synchronisation of gradients and parameters on both CPU and GPU backends. For
clusters that span multiple machines, libraries such as Horovod offer
additional orchestration and elastic scaling.

The `distributed_training.DistributedTrainer` helper spawns one worker per
process with `torch.multiprocessing.spawn`. Each worker initialises a
distributed process group via `init_distributed`, trains locally and then
averages synapse weights using `torch.distributed.all_reduce`. The averaged
weights are written back to every worker so all models remain in sync without
altering the core learning logic. `DistributedTrainer` accepts a configurable
`world_size` and `backend` (``gloo`` by default) allowing the same interface to
scale from a single machine to a small cluster.

Use ``dataset_replication.replicate_dataset`` to push training files to all
machines before starting jobs so that each worker loads identical data shards.

Workers require `init_distributed` to run before constructing `Core` instances.
Set `MASTER_ADDR` and `MASTER_PORT` environment variables when launching across
machines. Datasets can be synchronised by combining `DatasetCacheServer` with
the `dataset_replication` helpers so each rank pulls the same shards. After
training, metrics can be visualised live through `metrics_dashboard` when all
workers log to a shared directory.

Always call `cleanup_distributed()` on every worker once training completes to
ensure process groups are dismantled cleanly and resources are released.

Monitor resource usage on each node with `system_metrics.profile_resource_usage`
or run a long‑term `usage_profiler.UsageProfiler` instance to capture CPU, RAM
and GPU trends while distributed jobs execute.

### Remote hardware plugins

For heterogeneous clusters, a remote hardware plugin can expose accelerators or
specialised hardware. Set ``remote_hardware.tier_plugin`` in ``config.yaml`` to
an import path that returns a remote tier object. The built-in
``GrpcRemoteTier`` communicates with a gRPC service and serves as a reference
implementation. See [public_api.md](public_api.md#remote-hardware-plugins) for
API details.
