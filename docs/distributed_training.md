# Distributed Training Approaches

The Marble framework supports running Neuronenblitz models across multiple
processes using PyTorch's DistributedDataParallel (DDP). DDP provides efficient
synchronisation of gradients and parameters on both CPU and GPU backends. For
clusters that span multiple machines, libraries such as Horovod offer
additional orchestration and elastic scaling.

The provided `DistributedTrainer` wrapper initialises a Torch distributed
process group and averages synapse weights after each training batch. This
ensures all workers remain in sync without introducing significant changes to
the core algorithms.
