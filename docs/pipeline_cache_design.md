# Pipeline Step Result Caching

## Overview
The pipeline now supports persisting step outputs to disk. Providing a
`cache_dir` to `Pipeline.execute` stores the result of every executed step in
the specified directory. On subsequent runs the pipeline hashes the step's
specification (function, parameters and dependencies) to determine whether a
cached result exists and reuses it if available.

## GPU and CPU Handling
Results are serialised with `torch.save` and restored with `torch.load` using
`map_location` so tensors automatically move to the current execution device.
This ensures cached data seamlessly migrates between CPU and GPU environments.

## File Layout
Cache files are named `<index>_<name>_<hash>.pt` where `<hash>` is a SHA256
checksum of the step specification. Each file contains the raw Python object
returned by the step.

## Hooks and Side Effects
Pre-hooks always run even when a cached result is loaded. Post-hooks operate on
the loaded result allowing downstream consumers such as training loops to
execute unchanged.
