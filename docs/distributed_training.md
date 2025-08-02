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
