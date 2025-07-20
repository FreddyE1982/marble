# MARBLE Step-by-Step Tutorial

This tutorial demonstrates every major component of MARBLE through a series of projects. Each project builds on the previous one and introduces new functionality. By the end you will have explored all options listed in `CONFIGURABLE_PARAMETERS.md` and documented in detail in `yaml-manual.txt`.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Review the architecture overview in [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) and the configuration reference in [yaml-manual.txt](yaml-manual.txt).

## Project 1 – Numeric Regression (Easy)

**Goal:** Train MARBLE on a simple numeric dataset.

1. Download the [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv).
2. Convert each row into `(input, target)` pairs where `input` is the vector of features and `target` is the quality score.
3. Load `config.yaml` and adjust parameters under `core` if you wish to change representation size or initial seed resolution.
4. Create a `MARBLE` instance:
   ```python
   from marble_main import MARBLE
   from config_loader import load_config

   cfg = load_config()
   marble = MARBLE(cfg["core"])
   ```
5. Train:
   ```python
   marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
   ```
6. Inspect live metrics with the `MetricsVisualizer` which plots loss and memory usage. Options such as `fig_width` and `color_scheme` are configurable in `config.yaml` under `metrics_visualizer`.
7. To slowly reduce regularization as training progresses, set `dropout_probability` and `dropout_decay_rate` under `neuronenblitz` in `config.yaml`. A decay rate below `1.0` multiplies the current dropout after each epoch.

This project introduces the **Core**, **Neuronenblitz** and **Brain** objects along with the data compression pipeline.

## Project 2 – Image Classification (Medium)

**Goal:** Use the built-in asynchronous training and evolutionary tools on an image dataset.

1. Download the [CIFAR‑10 images](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and unpack them.
2. Construct `(input, target)` pairs by reading each image array and its label.
3. Enable asynchronous training by calling `Brain.start_training(...)` and continue with `Brain.wait_for_training()` when finished. This allows inference with `dynamic_wander()` while training runs in a background thread.
4. Experiment with evolutionary functions:
   ```python
   mutated, pruned = marble.brain.evolve(mutation_rate=0.02, prune_threshold=0.05)
   ```
5. Set `dream_enabled: true` in the configuration to let the system consolidate memory in the background. Parameters like `dream_num_cycles` and `dream_interval` control how often dreaming occurs.

This project makes use of **asynchronous training**, **dreaming**, and the **evolutionary mechanisms** such as mutation and pruning.

## Project 3 – Remote Offloading (Harder)

**Goal:** Run part of the brain on a different machine.

1. On the remote machine, start a server:
   ```python
   from remote_offload import RemoteBrainServer
   server = RemoteBrainServer(port=8000)
   server.start()
   ```
2. On the training machine, create a `RemoteBrainClient` and pass it to both `Neuronenblitz` and `Brain`:
   ```python
   from remote_offload import RemoteBrainClient
   client = RemoteBrainClient("http://remote_host:8000")
   marble = MARBLE(cfg["core"], remote_client=client)
   ```
3. Enable offloading via `brain.offload_enabled = True` and call `brain.offload_high_attention(threshold=0.5)` to migrate heavily used lobes.
4. The same mechanism works with the torrent client. Configure `torrent_client` in the YAML file and call `brain.offload_high_attention()`.

Remote offloading demonstrates **RemoteBrainServer**, **RemoteBrainClient** and the optional torrent‑based distribution.

## Project 4 – Autograd and PyTorch Challenge (Advanced)

**Goal:** Combine MARBLE with a PyTorch model and compare results.

1. Obtain the [Scikit‑learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) using `load_digits()`.
2. Use `pytorch_challenge.load_pretrained_model()` to get a small SqueezeNet model.
3. Call `pytorch_challenge.run_challenge()` which trains MARBLE and the PyTorch model side by side while adjusting neuromodulatory stress when MARBLE performs worse.
4. Alternatively wrap the brain with `MarbleAutogradLayer` to apply PyTorch autograd directly:
   ```python
   from marble_autograd import MarbleAutogradLayer
   layer = MarbleAutogradLayer(marble.brain)
   out = layer(torch.tensor(1.0, requires_grad=True))
   ```

This project covers **autograd integration** and the **PyTorch challenge** mechanism.

## Project 5 – GPT Training (Expert)

**Goal:** Train a tiny language model inside MARBLE.

1. Download the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
2. Set the `gpt.enabled` section in `config.yaml` to `true` and provide the `dataset_path` pointing to the downloaded text file.
3. Run `advanced_gpt.load_text_dataset()` to tokenize the text and then `advanced_gpt.train_advanced_gpt()` to train the transformer defined in `advanced_gpt.py`.
4. Examine the generated text using the trained GPT by feeding starting tokens to the model.
5. Optionally apply **knowledge distillation** by training a student brain with `DistillationTrainer`, using a previously saved MARBLE model as the teacher.

This final project introduces the **GPT components**, **distillation**, and the **dimensional search** capability if `dimensional_search.enabled` is set in the configuration.
## Project 6 – Reinforcement Learning (Master)

**Goal:** Solve a simple GridWorld using Q-learning built on top of MARBLE.

1. Enable `reinforcement_learning.enabled` in `config.yaml` and also set
   `core.reinforcement_learning_enabled` and
   `neuronenblitz.reinforcement_learning_enabled` to `true`.
2. Either drive the environment manually using the new `rl_` methods on the
   core and Neuronenblitz or call `reinforcement_learning.train_gridworld()`
   which demonstrates these helpers in action.
3. Observe the total rewards returned after each episode to verify learning progress.


## Where to Go Next

The configuration file exposes many additional parameters covering memory management, neuromodulation, meta‑controller behaviour and more. Consult `CONFIGURABLE_PARAMETERS.md` for the complete list and see `yaml-manual.txt` for thorough descriptions such as the excerpt below:
```yaml
The configuration YAML file controls all components of MARBLE.  Each top-level
section corresponds to a subsystem and exposes parameters that can be tuned to
alter its behaviour.
```

Experiment by modifying these options and combining features from multiple projects. The test suite (`pytest`) exercises every component and can be run to verify your setup.
