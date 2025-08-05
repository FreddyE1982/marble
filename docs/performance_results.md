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
