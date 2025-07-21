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

1. **Download the data** using `wget`:
   ```bash
   wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
   ```
2. **Prepare the dataset** by loading the CSV with `pandas` and converting rows into `(input, target)` pairs where `input` contains all feature columns and `target` is the quality score.
3. **Split the data** into training and validation sets using `train_test_split` from `sklearn.model_selection` so that the training loop can monitor validation loss.
4. **Edit configuration**. Open `config.yaml` and modify values under `core` to adjust the representation size and other parameters. Save the file when done.
5. **Create a MARBLE instance**:
   ```python
   from marble_main import MARBLE
   from config_loader import load_config

   cfg = load_config()
   marble = MARBLE(cfg["core"])
   ```
6. **Train the model**:
   ```python
   marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
   ```
7. **Monitor progress** with the `MetricsVisualizer` which plots loss and memory usage. The `fig_width` and `color_scheme` options in `config.yaml` under `metrics_visualizer` control the visual appearance.
8. **Gradually reduce regularization** by setting `dropout_probability` and `dropout_decay_rate` under `neuronenblitz`. A decay rate below `1.0` multiplies the current dropout after each epoch.

This project introduces the **Core**, **Neuronenblitz** and **Brain** objects along with the data compression pipeline.

## Project 2 – Image Classification (Medium)

**Goal:** Use the built-in asynchronous training and evolutionary tools on an image dataset.

1. **Download the dataset**:
   ```bash
   wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   tar -xzf cifar-10-python.tar.gz
   ```
2. **Create training pairs** by loading each image array from the extracted files and pairing it with the provided label to form `(input, target)` tuples. Normalise the image values to the range `[0, 1]`.
3. **Start asynchronous training** by calling:
   ```python
   marble.brain.start_training(train_examples, epochs=20, validation_examples=val_examples)
   ```
   You can continue executing code while training occurs in a background thread. When ready to wait for completion call `marble.brain.wait_for_training()`.
4. **Run inference concurrently** with `marble.brain.dynamic_wander(sample)` to test the partially trained network while training is still running.
5. **Experiment with evolutionary functions** to mutate or prune synapses:
   ```python
   mutated, pruned = marble.brain.evolve(mutation_rate=0.02, prune_threshold=0.05)
   ```
6. **Enable dreaming** by setting `dream_enabled: true` in `config.yaml`. Parameters like `dream_num_cycles` and `dream_interval` determine how often memory consolidation happens in the background.

This project makes use of **asynchronous training**, **dreaming**, and the **evolutionary mechanisms** such as mutation and pruning.

## Project 3 – Remote Offloading (Harder)

**Goal:** Run part of the brain on a different machine.

1. **Start the remote server** on another machine:
   ```python
   from remote_offload import RemoteBrainServer
   server = RemoteBrainServer(port=8000)
   server.start()
   ```
   Make sure this machine has the same dependencies installed so the brain lobes can be executed remotely.
2. **Create a remote client** on your training machine and pass it to MARBLE:
   ```python
   from remote_offload import RemoteBrainClient
   client = RemoteBrainClient("http://remote_host:8000")
   marble = MARBLE(cfg["core"], remote_client=client)
   ```
3. **Enable offloading** by setting `marble.brain.offload_enabled = True` and then call:
   ```python
   marble.brain.offload_high_attention(threshold=0.5)
   ```
   This migrates the most heavily used lobes to the remote machine.
4. **Use the torrent client** in the same way by configuring the `torrent_client` section of `config.yaml` and calling `marble.brain.offload_high_attention()` to distribute lobes through peer‑to‑peer transfer.

Remote offloading demonstrates **RemoteBrainServer**, **RemoteBrainClient** and the optional torrent‑based distribution.

## Project 4 – Autograd and PyTorch Challenge (Advanced)

**Goal:** Combine MARBLE with a PyTorch model and compare results.

1. **Load the digits dataset** using `load_digits()` from `sklearn.datasets` and flatten the image arrays so each sample becomes a vector.
2. **Retrieve a pretrained PyTorch network** by calling `pytorch_challenge.load_pretrained_model()` which returns a lightweight SqueezeNet model for comparison.
3. **Run the challenge** with
   ```python
   pytorch_challenge.run_challenge(digits, pretrained_model=pretrained, cfg=cfg)
   ```
   This trains MARBLE and the PyTorch model side by side while increasing neuromodulatory stress whenever MARBLE performs worse.
4. **Direct autograd integration** is possible by wrapping the brain with `MarbleAutogradLayer` so PyTorch optimizers can be applied directly:
   ```python
   from marble_autograd import MarbleAutogradLayer
   layer = MarbleAutogradLayer(marble.brain)
   out = layer(torch.tensor(1.0, requires_grad=True))
   ```

This project covers **autograd integration** and the **PyTorch challenge** mechanism.

## Project 5 – GPT Training (Expert)

**Goal:** Train a tiny language model inside MARBLE.

1. **Download the dataset** and place it in a `data/` directory:
   ```bash
   mkdir -p data
   wget -O data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```
2. **Enable the GPT components** by editing `config.yaml` and setting `gpt.enabled: true`. Also provide `dataset_path: data/tinyshakespeare.txt` in that section.
3. **Tokenize and train** using the helper functions:
   ```python
   dataset = advanced_gpt.load_text_dataset(cfg["gpt"]["dataset_path"])
   advanced_gpt.train_advanced_gpt(marble.brain, dataset, epochs=5)
   ```
4. **Generate text** once training completes by calling `advanced_gpt.generate_text(marble.brain, "Once upon a time")`.
5. **Optionally distill** the knowledge to a smaller network with `DistillationTrainer` by loading a saved model and training a student brain against it.

This final project introduces the **GPT components**, **distillation**, and the **dimensional search** capability if `dimensional_search.enabled` is set in the configuration.
## Project 6 – Reinforcement Learning (Master)

**Goal:** Solve a simple GridWorld using Q-learning built on top of MARBLE.

1. **Enable reinforcement learning** by editing `config.yaml` and setting `reinforcement_learning.enabled: true`. Also set
   `core.reinforcement_learning_enabled` and
   `neuronenblitz.reinforcement_learning_enabled` to `true` so that all components are prepared for Q‑learning updates.
2. **Run the built-in GridWorld example** with:
   ```python
   from reinforcement_learning import train_gridworld
   history = train_gridworld(marble.brain, episodes=50)
   ```
   This uses helper functions that drive the environment and update the Q-table stored inside the Neuronenblitz object.
3. **Check rewards** in `history` after each episode to verify that the policy improves over time.

## Project 7 – Contrastive Learning (Expert+)

**Goal:** Learn robust representations without labels.

1. **Download and extract** the [STL‑10 dataset](https://ai.stanford.edu/~acoates/stl10/) using `wget` followed by `tar -xzf` to obtain the unlabeled split.
2. **Enable the contrastive learner** by setting `contrastive_learning.enabled: true` in `config.yaml` and choose a `batch_size` that fits your GPU memory.
3. **Define data augmentations** such as random cropping and horizontal flipping, then call `Neuronenblitz.contrastive_train(images, augment_fn)` to learn representations from the unlabeled images.
4. **Fine-tune on labels** by reusing the trained weights and invoking the standard `train()` method on a labeled subset of STL‑10.

This project demonstrates the new **ContrastiveLearner** and how it integrates
with the existing Core and Neuronenblitz components.

## Project 8 – Hebbian Learning (Research)

**Goal:** Explore unsupervised Hebbian updates integrated with Neuronenblitz.

1. **Turn on Hebbian learning** by editing `config.yaml` and setting the `hebbian_learning` section to enable the feature while providing values for `learning_rate` and `weight_decay`.
2. **Create the learner**:
   ```python
   from hebbian_learning import HebbianLearner
   learner = HebbianLearner(core, neuronenblitz)
   ```
3. **Train on unlabeled data** by passing a list of input vectors to `learner.train(inputs)`.
4. **Review synapse adjustments** in `learner.history` to see how correlations between activations strengthen or weaken connections.



## Project 9 – Adversarial Learning (Cutting Edge)

**Goal:** Train a generator and discriminator using Neuronenblitz.

1. **Activate adversarial mode** by setting `adversarial_learning.enabled: true` in `config.yaml` and specify `epochs`, `batch_size` and the latent `noise_dim` used by the generator.
2. **Create the networks** by instantiating two `Neuronenblitz` objects that share the same Core: one acts as generator and the other as discriminator.
3. **Construct an `AdversarialLearner`** with these networks and call `learner.train(real_values)` to alternate generator and discriminator updates.
4. **Sample new data** after training by passing random noise vectors to the generator's `dynamic_wander` method.

## Project 10 – Autoencoder Learning (Frontier)

**Goal:** Reconstruct noisy inputs using an autoencoder built with Neuronenblitz.

1. **Enable the autoencoder module** by setting `autoencoder_learning.enabled: true` in `config.yaml` and choose values for `epochs`, `batch_size` and `noise_std` which controls how much noise is added during training.
2. **Instantiate the classes**:
   ```python
   from autoencoder_learning import AutoencoderLearner
   auto = AutoencoderLearner(core, neuronenblitz)
   ```
3. **Train** using `auto.train(values)` where `values` is a list of numeric vectors. The learner corrupts inputs with Gaussian noise of standard deviation `noise_std` and learns to reconstruct the originals.
4. **Inspect progress** in `auto.history` to see reconstruction losses decreasing over epochs.

## Project 11 – Semi-Supervised Learning (Frontier)

**Goal:** Combine labeled and unlabeled data using the `SemiSupervisedLearner`.

1. **Enable the module** by editing `config.yaml` and setting `semi_supervised_learning.enabled: true`. Configure `epochs`, `batch_size` and the weight `unlabeled_weight` applied to the unsupervised loss.
2. **Create the learner** with your existing network:
   ```python
   from semi_supervised_learning import SemiSupervisedLearner
   learner = SemiSupervisedLearner(core, neuronenblitz)
   ```
3. **Train** using two lists: one containing `(input, target)` pairs and the other containing unlabeled inputs. Call `learner.train(labeled, unlabeled)`.
4. **Inspect history** for both supervised and consistency losses recorded in `learner.history` to gauge progress.

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

1. **Enable federated mode** by setting `federated_learning.enabled: true` in `config.yaml`. Define the number of communication `rounds` and how many `local_epochs` each client trains before averaging.
2. **Instantiate clients**: create a separate `Core` and `Neuronenblitz` for each participant and pass these to a `FederatedAveragingTrainer` instance.
3. **Train round by round** by providing a list of datasets to `trainer.train_round(client_data)` for each round. The trainer aggregates parameters after every local training phase.
4. **Check synchronisation** by comparing synapse weights across clients after training to ensure the averaging step worked.


## Project 13 – Curriculum Learning (Frontier)

**Goal:** Gradually introduce harder examples to improve stability.**

1. **Turn on curriculum learning** by setting `curriculum_learning.enabled: true` in `config.yaml` and provide a list under `schedule` describing when to introduce more difficult samples.
2. **Create the learner** using your network:
   ```python
   from curriculum_learning import CurriculumLearner
   learner = CurriculumLearner(core, neuronenblitz)
   ```
3. **Order your dataset** by difficulty and call `learner.train(sorted_samples)` to progressively feed in harder examples.
4. **Track progress** via `learner.history`, which records the loss as each stage of the curriculum is reached.

## Project 14 – Meta Learning (Frontier)

**Goal:** Adapt quickly to new tasks using the Reptile algorithm.**

1. **Activate meta learning** in `config.yaml` by setting `meta_learning.enabled: true` and specify `epochs`, `inner_steps` and a meta learning rate `meta_lr`.
2. **Prepare tasks**: each task should provide its own list of `(input, target)` pairs. Gather them into a list named `tasks`.
3. **Create the meta learner**:
   ```python
   from meta_learning import MetaLearner
   learner = MetaLearner(core, neuronenblitz)
   ```
4. **Iterate over epochs** calling `learner.train_step(tasks)` each time to perform the Reptile update.
5. **Review meta-loss** from `learner.history` to see how quickly the model adapts across tasks.

## Project 15 – Transfer Learning (Frontier)

**Goal:** Fine-tune a pretrained MARBLE model on a new dataset while freezing
a subset of synapses.**

1. **Enable transfer learning** by setting `transfer_learning.enabled: true` in `config.yaml`. Choose `epochs`, `batch_size` and a `freeze_fraction` specifying what portion of synapses remain unchanged.
2. **Create the transfer learner** after loading an existing model (or training a base model) and passing the `Core` and `Neuronenblitz` objects to `TransferLearner`.
3. **Fine-tune** on a new dataset by calling `learner.train(new_pairs)` where `new_pairs` is your list of `(input, target)` examples.
4. **Tune `freeze_fraction`** to control how many synapses stay fixed during fine‑tuning and monitor `learner.history` to check performance on the new task.

## Project 16 – Continual Learning (Frontier)

**Goal:** Train sequential tasks while replaying previous examples.**

1. **Enable continual learning** in the configuration by setting `continual_learning.enabled: true` and provide values for `epochs` and the replay `memory_size`.
2. **Create the learner**:
   ```python
   from continual_learning import ReplayContinualLearner
   learner = ReplayContinualLearner(core, neuronenblitz)
   ```
3. **Train sequentially**. For each new dataset that becomes available call `learner.train(data)`; the learner automatically mixes in examples from its replay memory.
4. **Monitor reconstruction loss** via `learner.history` to see how well the model retains previous knowledge across tasks.

## Project 17 – Imitation Learning (Exploration)

**Goal:** Learn a policy directly from demonstration pairs.**

1. **Enable imitation mode** by setting `imitation_learning.enabled: true` in the configuration and choose `epochs` along with the `max_history` size that limits how many demonstrations are stored.
2. **Create the learner**:
   ```python
   from imitation_learning import ImitationLearner
   imitator = ImitationLearner(core, neuronenblitz)
   ```
3. **Record demonstrations** using `imitator.record(input, action)` for each step of the task, then call `imitator.train()` (or `neuronenblitz.imitation_train()`) to learn from the history.
4. **Evaluate** the cloned policy by passing new inputs to `dynamic_wander` and observing the predicted actions.

## Project 18 – Harmonic Resonance Learning (Novel)

**Goal:** Explore the experimental frequency-based paradigm.**

1. **Enable harmonic resonance** by editing the `harmonic_resonance_learning` section of `config.yaml` and setting `enabled: true` with parameters `epochs`, `base_frequency` and `decay`.
2. **Instantiate the learner**:
   ```python
   from harmonic_resonance_learning import HarmonicResonanceLearner
   learner = HarmonicResonanceLearner(core, neuronenblitz)
   ```
3. **Train** by repeatedly calling `learner.train_step(value, target)` for the specified number of epochs.
4. **Observe frequency error** in `learner.history` to understand how phase alignment evolves.

## Project 19 – Synaptic Echo Learning (Novel)

**Goal:** Experiment with echo-modulated weight updates.**

1. **Enable synaptic echo** by setting `synaptic_echo_learning.enabled: true` in the YAML configuration and choose values for `epochs` and `echo_influence`.
2. **Instantiate the learner** ensuring the underlying Neuronenblitz has `use_echo_modulation=True`:
   ```python
   from synaptic_echo_learning import SynapticEchoLearner
   learner = SynapticEchoLearner(core, neuronenblitz)
   ```
3. **Train repeatedly** with `learner.train_step(value, target)` to apply the echo-modulated updates.
4. **Inspect** both `learner.history` and the synapse echo buffers in the Neuronenblitz object to see how past activations influence current learning.

## Project 20 – Fractal Dimension Learning (Novel)

**Goal:** Let MARBLE expand its representations when activity becomes complex.**

1. **Enable fractal dimension learning** by setting `fractal_dimension_learning.enabled: true` in the configuration and choose `epochs` along with the desired `target_dimension`.
2. **Instantiate the learner**:
   ```python
   from fractal_dimension_learning import FractalDimensionLearner
   learner = FractalDimensionLearner(core, neuronenblitz)
   ```
3. **Train** with `learner.train(pairs)` where `pairs` are the usual `(input, target)` tuples.
4. **Watch representation size** via `core.rep_size` or `learner.history` to see when new dimensions are added.

## Project 21 – Quantum Flux Learning (Novel)

**Goal:** Explore phase-modulated weight updates.**

1. **Enable quantum flux learning** in the configuration by setting `quantum_flux_learning.enabled: true` and choose values for `epochs` and the update `phase_rate`.
2. **Create the learner**:
   ```python
   from quantum_flux_learning import QuantumFluxLearner
   learner = QuantumFluxLearner(core, neuronenblitz)
   ```
3. **Train** by repeatedly calling `learner.train_step(input, target)` which applies phase-modulated updates to the synapses.
4. **Track phases** in `learner.phases` to understand how the system evolves over time.

## Project 22 – Dream Reinforcement Synergy (Novel)

**Goal:** Combine dreaming with reinforcement-like updates.**

1. **Enable dream reinforcement** by setting `dream_reinforcement_learning.enabled: true` in the YAML file and configure `episodes`, `dream_cycles` and `dream_strength`.
2. **Instantiate the learner**:
   ```python
   from dream_reinforcement_learning import DreamReinforcementLearner
   learner = DreamReinforcementLearner(core, neuronenblitz)
   ```
3. **Train episodes** by repeatedly calling `learner.train_episode(input, target)` for every interaction step.
4. **Imaginary updates** occur after each real step; `dream_cycles` controls how many of these dreaming iterations happen.

## Project 23 – Omni Learning Paradigm (Advanced)

**Goal:** Train using every supported paradigm at once.**

1. **Combine cores** by creating several `Core` objects and merging them with `interconnect_cores` from `core_interconnect`.
2. **Instantiate one Neuronenblitz** using the combined core so the learner sees a unified network.
3. **Create an `OmniLearner`** with the merged core and single Neuronenblitz instance.
4. **Train** by providing a list of `(input, target)` samples and calling `learner.train(data, epochs=5)`.
5. **All paradigms run sequentially**, leveraging interconnection synapses so multiple cores behave as one integrated system.
