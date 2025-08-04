# Macro Step Design

Macro steps allow several pipeline operations to be grouped and executed as a
single logical step.  Each macro defines a list of regular step specifications
which are run sequentially on the currently active device.  The pipeline treats
a macro like any other step so dependencies, caching and pre/post hooks all
apply uniformly.

## Execution model

1. Each sub-step inside the macro is validated against the pipeline schema.
2. During `Pipeline.execute` a sub-pipeline is constructed from the macro's
   steps and executed with the same metrics visualiser, logging and debugging
   configuration as the parent pipeline.
3. Results from the sub-pipeline are returned as a list and can be cached to
disk.  Cached macro results embed the sub-step specifications so modifying any
sub-step invalidates the cache.

Macros are device agnostic.  When a CUDA device is available the sub-pipeline
runs on the GPU; otherwise it runs on the CPU.  Mixed CPU/GPU macro pipelines
are also supported.
