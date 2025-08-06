# Performance Benchmarks

## Asynchronous Pipeline Execution

Measured execution times for a small pipeline of three artificial steps:

| Device | Synchronous (s) | Asynchronous (s) |
|--------|-----------------|------------------|
| CPU    | 0.003 | 0.001 |
| GPU    | N/A | N/A |

## Pipeline Cache Stress Test

Ratio of second run time to first run time for five cached steps:

| Device | Second/First Time Ratio |
|--------|------------------------|
| CPU    | 0 additional calls |
| GPU    | N/A |

## Neuronenblitz Parallel Workers

Benchmark comparing single vs two worker threads training 200 examples:

| Workers | Time (s) | Throughput (examples/s) |
|---------|---------:|-----------------------:|
| 1       | 0.0027   | 74853.55 |
| 2       | 0.0140   | 14243.54 |

While the small benchmark shows limited benefit from two workers due to
threading overhead, larger workloads with heavier `train_example` logic can
observe notable speedups on multi-core CPUs.
