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

## Project 7 – Contrastive Learning (Expert+)

**Goal:** Learn robust representations without labels.

1. Download the [STL-10 dataset](https://ai.stanford.edu/~acoates/stl10/) and
   extract the unlabeled split.
2. Enable `contrastive_learning.enabled` in `config.yaml` and set an appropriate
   `batch_size` for your hardware.
3. Create a simple augmentation function (e.g. random crop and flip) and pass it
   to `Neuronenblitz.contrastive_train()` together with the images.
4. After training, reuse the learned weights in a supervised task by fine-tuning
   with the regular `train()` method.

This project demonstrates the new **ContrastiveLearner** and how it integrates
with the existing Core and Neuronenblitz components.

## Project 8 – Hebbian Learning (Research)

**Goal:** Explore unsupervised Hebbian updates integrated with Neuronenblitz.

1. Enable `hebbian_learning` settings in `config.yaml` to configure `learning_rate` and `weight_decay`.
2. Instantiate `HebbianLearner` with the existing Core and Neuronenblitz objects.
3. Call `HebbianLearner.train()` on a list of inputs without targets.
4. Inspect weight changes along visited paths to verify correlation-based updates.



## Project 9 – Adversarial Learning (Cutting Edge)

**Goal:** Train a generator and discriminator using Neuronenblitz.

1. Enable `adversarial_learning` in `config.yaml` and configure `epochs`, `batch_size` and `noise_dim`.
2. Instantiate two `Neuronenblitz` objects sharing the same Core.
3. Create an `AdversarialLearner` with these objects.
4. Call `AdversarialLearner.train()` on a list of real numeric values.
5. Generate new samples by passing random noise to `dynamic_wander` of the generator.

## Project 10 – Autoencoder Learning (Frontier)

**Goal:** Reconstruct noisy inputs using an autoencoder built with Neuronenblitz.

1. Set `autoencoder_learning.enabled` to `true` in `config.yaml` and adjust `epochs`, `batch_size` and `noise_std` as desired.
2. Instantiate a `Neuronenblitz` network and an `AutoencoderLearner`.
3. Call `AutoencoderLearner.train()` on your dataset of numeric values.
4. Inspect `learner.history` to see reconstruction losses.

## Project 11 – Semi-Supervised Learning (Frontier)

**Goal:** Combine labeled and unlabeled data using the `SemiSupervisedLearner`.

1. Enable `semi_supervised_learning.enabled` in `config.yaml` and set `epochs`, `batch_size` and `unlabeled_weight`.
2. Instantiate a `Neuronenblitz` network and a `SemiSupervisedLearner`.
3. Provide matching lists of labeled pairs and unlabeled inputs to `train()`.
4. Review `learner.history` for supervised and consistency loss values.

## Where to Go Next

The configuration file exposes many additional parameters covering memory management, neuromodulation, meta‑controller behaviour and more. Consult `CONFIGURABLE_PARAMETERS.md` for the complete list and see `yaml-manual.txt` for thorough descriptions such as the excerpt below:
```yaml
The configuration YAML file controls all components of MARBLE.  Each top-level
section corresponds to a subsystem and exposes parameters that can be tuned to
alter its behaviour.
```

Experiment by modifying these options and combining features from multiple projects. The test suite (`pytest`) exercises every component and can be run to verify your setup.

## Project 12 – Federated Learning (Frontier)

**Goal:** Train multiple Neuronenblitz networks on separate datasets and combine them using federated averaging.**

1. Enable `federated_learning.enabled` in `config.yaml` and set `rounds` and `local_epochs`.
2. Create one `Core`/`Neuronenblitz` pair for each client and instantiate `FederatedAveragingTrainer` with them.
3. Provide a dataset list matching the clients to `train_round()` for each communication round.
4. After training, examine synapse weights to confirm they were synchronised across clients.


## Project 13 – Curriculum Learning (Frontier)

**Goal:** Gradually introduce harder examples to improve stability.**

1. Enable `curriculum_learning.enabled` in `config.yaml` and set `epochs` and `schedule`.
2. Instantiate a `Neuronenblitz` network and a `CurriculumLearner`.
3. Provide your dataset of `(input, target)` pairs ordered by a difficulty function.
4. Call `CurriculumLearner.train()` to run through the curriculum.
5. Monitor `learner.history` for loss values as harder samples are introduced.

## Project 14 – Meta Learning (Frontier)

**Goal:** Adapt quickly to new tasks using the Reptile algorithm.**

1. Enable `meta_learning.enabled` in `config.yaml` and configure `epochs`, `inner_steps` and `meta_lr`.
2. Create a list of tasks, where each task is a list of `(input, target)` pairs.
3. Instantiate a `MetaLearner` with your `Core` and `Neuronenblitz` objects.
4. Call `MetaLearner.train_step(tasks)` inside a loop for the desired number of epochs.
5. Inspect `learner.history` to track the average meta-loss across tasks.

## Project 15 – Transfer Learning (Frontier)

**Goal:** Fine-tune a pretrained MARBLE model on a new dataset while freezing
a subset of synapses.**

1. Enable `transfer_learning.enabled` in `config.yaml` and set `epochs`,
   `batch_size` and `freeze_fraction`.
2. Load an existing model or train a base model, then create a
   `TransferLearner` with the `Core` and `Neuronenblitz` objects.
3. Provide your new `(input, target)` pairs to `TransferLearner.train()`.
4. Adjust `freeze_fraction` to control how many synapses stay fixed during
   fine-tuning.

## Project 16 – Continual Learning (Frontier)

**Goal:** Train sequential tasks while replaying previous examples.**

1. Enable `continual_learning.enabled` in `config.yaml` and set `epochs` and
   `memory_size`.
2. Instantiate a `ReplayContinualLearner` with your `Core` and `Neuronenblitz`
   objects.
3. For each dataset that arrives over time, call `ReplayContinualLearner.train()`
   to update the model while old examples are replayed from memory.
4. Examine `learner.history` to monitor reconstruction loss across tasks.

## Project 17 – Imitation Learning (Exploration)

**Goal:** Learn a policy directly from demonstration pairs.**

1. Enable `imitation_learning.enabled` in `config.yaml` and set `epochs` and
   `max_history`.
2. Instantiate an `ImitationLearner` with your `Core` and `Neuronenblitz`
   objects.
3. Record demonstrations using `ImitationLearner.record(input, action)` then
   call `ImitationLearner.train()` or `Neuronenblitz.imitation_train()`.
4. Query `dynamic_wander` with new inputs to evaluate the cloned policy.

## Project 18 – Harmonic Resonance Learning (Novel)

**Goal:** Explore the experimental frequency-based paradigm.**

1. Enable `harmonic_resonance_learning.enabled` in `config.yaml` and set
   `epochs`, `base_frequency` and `decay` as desired.
2. Instantiate `HarmonicResonanceLearner` with your `Core` and `Neuronenblitz`
   objects.
3. Call `train_step(value, target)` in a loop for the specified number of epochs.
4. Inspect `learner.history` to monitor the frequency error over time.

## Project 19 – Synaptic Echo Learning (Novel)

**Goal:** Experiment with echo-modulated weight updates.**

1. Enable `synaptic_echo_learning.enabled` in `config.yaml` and set
   `epochs` and `echo_influence` as desired.
2. Instantiate `SynapticEchoLearner` with your `Core` and `Neuronenblitz`
   objects. The Neuronenblitz instance must have `use_echo_modulation`
   enabled.
3. Call `train_step(value, target)` repeatedly to train using the echo
   mechanism.
4. Examine `learner.history` and synapse echo buffers to understand how
   past activations influence learning.

## Project 20 – Fractal Dimension Learning (Novel)

**Goal:** Let MARBLE expand its representations when activity becomes complex.**

1. Enable `fractal_dimension_learning.enabled` in `config.yaml` and set `epochs` and `target_dimension`.
2. Create a `FractalDimensionLearner` with your `Core` and `Neuronenblitz` objects.
3. Provide `(input, target)` pairs to `FractalDimensionLearner.train()`.
4. Observe `core.rep_size` in `learner.history` to see when dimensions grow.

## Project 21 – Quantum Flux Learning (Novel)

**Goal:** Explore phase-modulated weight updates.**

1. Enable `quantum_flux_learning.enabled` in `config.yaml` and set `epochs` and `phase_rate`.
2. Instantiate `QuantumFluxLearner` using existing `Core` and `Neuronenblitz` instances.
3. Call `train_step(input, target)` repeatedly to apply flux-based updates.
4. Inspect `learner.phases` to understand how synapse phases evolve.

## Project 22 – Dream Reinforcement Synergy (Novel)

**Goal:** Combine dreaming with reinforcement-like updates.**

1. Enable `dream_reinforcement_learning.enabled` in `config.yaml` and set `episodes`, `dream_cycles` and `dream_strength`.
2. Instantiate `DreamReinforcementLearner` with your `Core` and `Neuronenblitz` objects.
3. Use `train_episode(input, target)` for each interaction step.
4. The `dream_cycles` parameter controls how many imaginary updates occur after each real one.

## Project 23 – Omni Learning Paradigm (Advanced)

**Goal:** Train using every supported paradigm at once.**

1. Create multiple `Core` objects and merge them using `interconnect_cores` from `core_interconnect`.
2. Instantiate a single `Neuronenblitz` with the combined core.
3. Create an `OmniLearner` using that core and neuronenblitz.
4. Provide a list of `(input, target)` samples and call `learner.train(data, epochs=5)`.
5. The learner sequentially executes all paradigms, leveraging interconnection synapses so the multiple cores behave like one.
