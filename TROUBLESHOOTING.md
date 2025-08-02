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

## Python Version Support
MARBLE officially supports Python 3.10 and above. The `scripts/convert_to_py38.py`
helper can backport the source tree for Python 3.8 or 3.9, but these runtimes are
not covered by the test suite and some optional features may be unavailable.

## Plugin Troubleshooting
* **Global workspace messages not appearing** – Ensure `global_workspace.enabled` is `true` in your configuration and that the plugin is activated before other plugins.
* **Attention codelets have no effect** – Verify that `attention_codelets.enabled` is `true` and that at least one codelet has been registered. Call `attention_codelets.run_cycle()` during training to broadcast proposals.
* **Remote hardware tier unavailable** – Ensure `remote_hardware.tier_plugin` points to a valid module and that the remote service is reachable from the network.
