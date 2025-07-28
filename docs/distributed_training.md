# Distributed Training Approaches

This note summarises common techniques for multi-GPU training that will guide future implementation.

## PyTorch Distributed Data Parallel (DDP)
DDP replicates the model across processes and synchronises gradients via efficient all-reduce operations. It is the standard approach in PyTorch and scales well across nodes. Launching typically uses `torchrun` or the `ddp` backend.

**Pros**
- Part of PyTorch, no extra dependencies
- Mature and widely used

**Cons**
- Requires careful setup of process groups

## Horovod
Horovod builds on MPI or NCCL to provide a unified API for distributed training across frameworks.

**Pros**
- Framework agnostic
- Can leverage efficient communication libraries

**Cons**
- Additional dependency
- Slightly more complex build steps

The implementation will start with DDP as a baseline, with hooks for Horovod if available.
