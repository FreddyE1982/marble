# Modules lacking CPU fallback tests

Currently all modules have CPU fallback tests in place. No outstanding modules require additional coverage.

## CPU-only test execution

The repository now includes `scripts/run_cpu_fallback_tests.py`, which runs each test module with `CUDA_VISIBLE_DEVICES` cleared so that execution occurs strictly on the CPU.  After running this script, append a summary of the results below to track ongoing verification runs.
