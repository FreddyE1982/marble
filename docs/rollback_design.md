# Rollback Design

The rollback mechanism restores the output of a previously executed pipeline
step and removes cached results of any subsequent steps.  This makes it easy to
undo failed experiments and continue iterating without recomputing earlier
stages.

## Behaviour

1. Steps are executed with caching enabled via `cache_dir`.
2. Calling `Pipeline.rollback(step_name, cache_dir)` locates the cached tensor
   produced by `step_name` and deletes all cached files for steps that follow.
3. The loaded result is returned for inspection and the pipeline can be
   re-executed starting from that step.  A CUDA device is used automatically
   when available; otherwise the data is loaded onto the CPU.

Rollback also removes any sub-cache directories created by macro steps so that
all downstream computations are recomputed on the next run.
