# Troubleshooting Guide

This guide lists common issues when running MARBLE and how to solve them.

## Installation Problems
* **Missing dependencies** – Ensure `pip install -r requirements.txt` completes without errors. Use a virtual environment to avoid permission issues.
* **GPU not detected** – Verify the CUDA toolkit is installed and PyTorch was compiled with CUDA support. Fall back to CPU by setting `CUDA_VISIBLE_DEVICES=""` if necessary.

## Runtime Errors
* **Invalid YAML** – Configuration files are validated on load. Check the error message for the offending key and consult `yaml-manual.txt` for parameter descriptions.
* **Out of memory** – Reduce `core.width` and `core.height` or lower the dataset size. Monitor system and GPU memory using the metrics dashboard.

## Training Instability
* **Loss becomes NaN** – This may be caused by extreme learning rates or numeric overflow. Enable gradient clipping in `config.yaml` and reduce `neuronenblitz.learning_rate`.
* **Early stopping triggers too soon** – Increase `brain.early_stopping_patience` or adjust `brain.early_stopping_delta`.

## Benchmarking Issues
* **Unexpectedly slow message passing** – Use the CLI option `--benchmark-msgpass` to measure throughput.
  A value above a few milliseconds per iteration on modern hardware may indicate
  that GPU acceleration is disabled. Check that `cupy` detects your GPU and that
  `torch.cuda.is_available()` returns `True`.

For further help open an issue on the project repository with your configuration and logs.
