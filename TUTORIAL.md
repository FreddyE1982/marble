# MARBLE Step-by-Step Tutorial

This tutorial demonstrates every major component of MARBLE through a series of projects. Each project builds on the previous one and introduces new functionality. The instructions below walk through every step in detail so that you can replicate the experiments exactly.

## Prerequisites

1. **Install dependencies** so that all modules are available. Run the following command inside the repository root and wait for the installation to finish:
   ```bash
   pip install -r requirements.txt
   ```
   After the packages are installed you can verify the environment by running `python -c "import marble_main"` which should finish silently.
2. **Review the documentation**. Read [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for a high level view of MARBLE and consult [yaml-manual.txt](yaml-manual.txt) for an explanation of every configuration option.

## Project 1 – Numeric Regression (Easy)

**Goal:** Train MARBLE on a simple numeric dataset.

1. **Download the data programmatically** so you have a local copy of the wine quality dataset:
   ```python
   import urllib.request

   url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
   urllib.request.urlretrieve(url, "winequality-red.csv")
   ```
   After the download completes you should see `winequality-red.csv` in the directory.
2. **Prepare the dataset** by loading the CSV with `pandas` and converting each row into `(input, target)` pairs. `input` contains the feature columns and `target` is the quality score:
   ```python
   import pandas as pd
   df = pd.read_csv('winequality-red.csv', sep=';')
   train_examples = [(row[:-1].to_numpy(), row[-1]) for row in df.to_numpy()]
   ```
3. **Split the data** into training and validation sets so the training loop can monitor validation loss:
   ```python
   from sklearn.model_selection import train_test_split
   train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
   ```
4. **Edit configuration**. Open `config.yaml` and modify the values under `core` to adjust the representation size and other parameters. Save the file after your edits.
   To experiment with sparser communication set `attention_dropout` to a value between `0.0` and `1.0`. Higher values randomly ignore more incoming messages during attention-based updates.
5. **Create a MARBLE instance** from the configuration:
   ```python
   from marble_main import MARBLE
   from config_loader import load_config

   cfg = load_config()
   marble = MARBLE(cfg['core'])
   ```
6. **Train the model** on the prepared data:
   ```python
   marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
   ```
7. **Monitor progress** with `MetricsVisualizer` which plots loss and memory usage. Adjust the `fig_width` and `color_scheme` options under `metrics_visualizer` in `config.yaml` to change the appearance.
8. **View metrics in your browser** by enabling `metrics_dashboard.enabled`. Set `window_size` to control the moving-average smoothing of the curves.
9. **Gradually reduce regularization** by setting `dropout_probability` and `dropout_decay_rate` under `neuronenblitz`. A decay rate below `1.0` multiplies the current dropout value after each epoch.

**Complete Example**
```python
# project1_numeric_regression.py
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from marble_main import MARBLE
from config_loader import load_config

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
urllib.request.urlretrieve(url, "winequality-red.csv")
df = pd.read_csv('winequality-red.csv', sep=';')
examples = [(row[:-1].to_numpy(), row[-1]) for row in df.to_numpy()]
train_examples, val_examples = train_test_split(examples, test_size=0.1, random_state=42)

cfg = load_config()
marble = MARBLE(cfg['core'])
marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
```
Run this script with `python project1_numeric_regression.py` to reproduce the first project end-to-end.

This project introduces the **Core**, **Neuronenblitz** and **Brain** objects along with the data compression pipeline.

## Project 2 – Image Classification (Medium)

**Goal:** Use the built-in asynchronous training and evolutionary tools on an image dataset.

1. **Download the dataset programmatically** so that you have the CIFAR‑10 archive locally:
   ```python
   import urllib.request, tarfile, os

   url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
   archive = "cifar-10-python.tar.gz"
   if not os.path.exists(archive):
       urllib.request.urlretrieve(url, archive)
       with tarfile.open(archive, "r:gz") as tar:
           tar.extractall()
   ```
   The extracted directory contains Python pickles for each data batch.
2. **Create training pairs** by loading each image array from the files and pairing it with the provided label to form `(input, target)` tuples. Normalise all pixel values into the range `[0, 1]` before continuing.
3. **Start asynchronous training** by calling:
   ```python
   marble.brain.start_training(train_examples, epochs=20, validation_examples=val_examples)
   ```
   Training now runs in a background thread so you can execute other code in the meantime. When you need to wait for completion call `marble.brain.wait_for_training()`.
4. **Run inference concurrently** with `marble.brain.dynamic_wander(sample)` to test the partially trained network while training is still running.
5. **Experiment with evolutionary functions** to mutate or prune synapses:
   ```python
   mutated, pruned = marble.brain.evolve(mutation_rate=0.02, prune_threshold=0.05)
   ```
   Mutations add noise to synapses while pruning removes the least useful ones.
6. **Enable dreaming** by setting `dream_enabled: true` in `config.yaml`. Parameters like `dream_num_cycles` and `dream_interval` determine how often memory consolidation happens in the background.

**Complete Example**
```python
# project2_image_classification.py
import urllib.request, tarfile, os, pickle
import numpy as np
from marble_main import MARBLE
from config_loader import load_config

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
archive = "cifar-10-python.tar.gz"
if not os.path.exists(archive):
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall()

def load_cifar_batches(path):
    data, labels = [], []
    for i in range(1, 6):
        with open(f"{path}/data_batch_{i}", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data.append(batch[b'data'])
        labels.extend(batch[b'labels'])
    data = np.vstack(data).reshape(-1, 3, 32, 32) / 255.0
    return list(zip(data, labels))

train_examples = load_cifar_batches('cifar-10-batches-py')
cfg = load_config()
marble = MARBLE(cfg['core'])
marble.brain.start_training(train_examples, epochs=20)
marble.brain.wait_for_training()
```
Running `python project2_image_classification.py` trains MARBLE on CIFAR‑10 with asynchronous execution.

This project makes use of **asynchronous training**, **dreaming**, and the **evolutionary mechanisms** such as mutation and pruning.

## Project 3 – Remote Offloading (Harder)

**Goal:** Run part of the brain on a different machine.

1. **Start the remote server** on another machine that has MARBLE installed:
   ```python
   from remote_offload import RemoteBrainServer
   server = RemoteBrainServer(port=8000)
   server.start()
   ```
   Ensure the remote machine has the same dependencies installed so the brain lobes can be executed remotely.
2. **Create a remote client** on your training machine and pass it when constructing MARBLE:
   ```python
   from remote_offload import RemoteBrainClient
   client = RemoteBrainClient('http://remote_host:8000')
   marble = MARBLE(cfg['core'], remote_client=client)
   ```
3. **Download a dataset** such as digits using `sklearn.datasets.load_digits()` for offloaded training:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   train_pairs = [(x, y) for x, y in zip(digits.data, digits.target)]
   ```
4. **Enable offloading** by setting `marble.brain.offload_enabled = True` and then call:
   ```python
   marble.brain.offload_high_attention(threshold=0.5)
   ```
   This migrates the most heavily used lobes to the remote machine.
5. **Use the torrent client** in the same way by configuring the `torrent_client` section of `config.yaml` and calling `marble.brain.offload_high_attention()` to distribute lobes through peer‑to‑peer transfer.

**Complete Example**
```python
# project3_remote_offloading.py
from remote_offload import RemoteBrainServer, RemoteBrainClient
from marble_main import MARBLE
from config_loader import load_config

# Run this on the remote machine
server = RemoteBrainServer(port=8000)
server.start()

# On the training machine
cfg = load_config()
client = RemoteBrainClient('http://remote_host:8000')
marble = MARBLE(cfg['core'], remote_client=client)
from sklearn.datasets import load_digits
digits = load_digits()
train_pairs = [(x, y) for x, y in zip(digits.data, digits.target)]
marble.brain.offload_enabled = True
marble.brain.offload_high_attention(threshold=0.5)
```
Execute the file on each machine as indicated to experiment with remote offloading.

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

**Complete Example**
```python
# project4_autograd_challenge.py
from sklearn.datasets import load_digits
from config_loader import load_config
from marble_main import MARBLE
from marble_autograd import MarbleAutogradLayer
import pytorch_challenge

digits = load_digits(return_X_y=True)
cfg = load_config()
marble = MARBLE(cfg['core'])
pretrained = pytorch_challenge.load_pretrained_model()
pytorch_challenge.run_challenge(digits, pretrained_model=pretrained, cfg=cfg)

layer = MarbleAutogradLayer(marble.brain)
out = layer(torch.tensor(1.0, requires_grad=True))
```
Run `python project4_autograd_challenge.py` to reproduce the integration with PyTorch.

This project covers **autograd integration** and the **PyTorch challenge** mechanism.

## Project 5 – GPT Training (Expert)

**Goal:** Train a tiny language model inside MARBLE.

1. **Download the dataset** and place it in a `data/` directory using Python:
   ```python
   import os, urllib.request

   os.makedirs('data', exist_ok=True)
   url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
   urllib.request.urlretrieve(url, 'data/tinyshakespeare.txt')
   ```
   The text file will be located at `data/tinyshakespeare.txt`.
2. **Enable the GPT components** by editing `config.yaml` and setting `gpt.enabled: true`. Also provide `dataset_path: data/tinyshakespeare.txt` in that section.
3. **Tokenize and train** using the helper functions:
   ```python
   dataset = advanced_gpt.load_text_dataset(cfg['gpt']['dataset_path'])
   advanced_gpt.train_advanced_gpt(marble.brain, dataset, epochs=5)
   ```
4. **Generate text** once training completes by calling `advanced_gpt.generate_text(marble.brain, 'Once upon a time')`.
5. **Optionally distill** the knowledge to a smaller network with `DistillationTrainer` by loading a saved model and training a student brain against it.

**Complete Example**
```python
# project5_gpt_training.py
from config_loader import load_config
from marble_main import MARBLE
import advanced_gpt
import os, urllib.request

cfg = load_config()
marble = MARBLE(cfg['core'])
os.makedirs('data', exist_ok=True)
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'data/tinyshakespeare.txt')
dataset = advanced_gpt.load_text_dataset(cfg['gpt']['dataset_path'])
advanced_gpt.train_advanced_gpt(marble.brain, dataset, epochs=5)
print(advanced_gpt.generate_text(marble.brain, 'Once upon a time'))
```
Run `python project5_gpt_training.py` after enabling GPT settings in `config.yaml`.

This final project introduces the **GPT components**, **distillation**, and the **dimensional search** capability if `dimensional_search.enabled` is set in the configuration. It also demonstrates the optional **n‑dimensional topology** feature controlled by `n_dimensional_topology.enabled`, which gradually expands the representation when learning stagnates.

## Project 6 – Reinforcement Learning (Master)

**Goal:** Solve a simple GridWorld using Q-learning built on top of MARBLE.

1. **Enable reinforcement learning** by editing `config.yaml` and setting `reinforcement_learning.enabled: true`. Also set `core.reinforcement_learning_enabled` and `neuronenblitz.reinforcement_learning_enabled` to `true` so that all components are prepared for Q‑learning updates.
2. **Download an expert trajectory dataset** using the Hugging Face `datasets` library:
   ```python
   from datasets import load_dataset
   expert = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
   ```
3. **Run the built-in GridWorld example** with:
   ```python
   from reinforcement_learning import train_gridworld
   history = train_gridworld(marble.brain, episodes=50)
   ```
   This uses helper functions that drive the environment and update the Q-table stored inside the Neuronenblitz object.
4. **Check rewards** in `history` after each episode to verify that the policy improves over time.

**Complete Example**
```python
# project6_reinforcement_learning.py
from config_loader import load_config
from marble_main import MARBLE
from reinforcement_learning import train_gridworld

cfg = load_config()
marble = MARBLE(cfg['core'])
from datasets import load_dataset
expert = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
history = train_gridworld(marble.brain, episodes=50)
print('Final reward:', history[-1])
```
Execute `python project6_reinforcement_learning.py` to run the built-in GridWorld.

## Project 7 – Contrastive Learning (Expert+)

**Goal:** Learn robust representations without labels.

1. **Download and extract** the [STL‑10 dataset](https://ai.stanford.edu/~acoates/stl10/) programmatically:
   ```python
   from torchvision.datasets import STL10
   import numpy as np

   unlabeled_ds = STL10(root='data', split='unlabeled', download=True)
   unlabeled = [np.asarray(img) / 255.0 for img, _ in unlabeled_ds]
   labeled_ds = STL10(root='data', split='train', download=True)
   labeled = [(np.asarray(img) / 255.0, label) for img, label in labeled_ds]
   ```
2. **Enable the contrastive learner** by setting `contrastive_learning.enabled: true` in `config.yaml` and choose a `batch_size` that fits your GPU memory.
3. **Define data augmentations** such as random cropping and horizontal flipping, then call `Neuronenblitz.contrastive_train(images, augment_fn)` to learn representations from the unlabeled images.
4. **Fine-tune on labels** by reusing the trained weights and invoking the standard `train()` method on a labeled subset of STL‑10.

**Complete Example**
```python
# project7_contrastive_learning.py
from config_loader import load_config
from marble_main import MARBLE
from marble_utils import augment_image

cfg = load_config()
marble = MARBLE(cfg['core'])
from torchvision.datasets import STL10
import numpy as np

unlabeled_ds = STL10(root='data', split='unlabeled', download=True)
unlabeled = [np.asarray(img) / 255.0 for img, _ in unlabeled_ds]
labeled_ds = STL10(root='data', split='train', download=True)
labeled = [(np.asarray(img) / 255.0, label) for img, label in labeled_ds]
marble.neuronenblitz.contrastive_train(unlabeled, augment_image)
marble.brain.train(labeled, epochs=5)
```
Run this script to reproduce the contrastive learning workflow.

This project demonstrates the new **ContrastiveLearner** and how it integrates with the existing Core and Neuronenblitz components.

## Project 8 – Hebbian Learning (Research)

**Goal:** Explore unsupervised Hebbian updates integrated with Neuronenblitz.

1. **Turn on Hebbian learning** by editing `config.yaml` and setting the `hebbian_learning` section to enable the feature while providing values for `learning_rate` and `weight_decay`.
2. **Create the learner**:
   ```python
   from hebbian_learning import HebbianLearner
   learner = HebbianLearner(core, neuronenblitz)
   ```
3. **Download real unlabeled data** using the [Fashion‑MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset:
   ```python
   from datasets import load_dataset
   ds = load_dataset('fashion_mnist', split='train')
   inputs = [x['image'].reshape(-1).numpy() / 255.0 for x in ds]
   ```
4. **Train** on these vectors with `learner.train(inputs)` and review `learner.history` to see how correlations strengthen or weaken connections.

**Complete Example**
```python
# project8_hebbian_learning.py
from config_loader import load_config
from marble_main import MARBLE
from hebbian_learning import HebbianLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = HebbianLearner(marble.core, marble.neuronenblitz)
from datasets import load_dataset
ds = load_dataset('fashion_mnist', split='train')
inputs = [x['image'].reshape(-1).numpy() / 255.0 for x in ds]
learner.train(inputs)
print(learner.history[-1])
```
Execute this file to observe Hebbian updates on your data.

## Project 9 – Adversarial Learning (Cutting Edge)

**Goal:** Train a generator and discriminator using Neuronenblitz.

1. **Activate adversarial mode** by setting `adversarial_learning.enabled: true` in `config.yaml` and specify `epochs`, `batch_size` and the latent `noise_dim` used by the generator.
2. **Create the networks** by instantiating two `Neuronenblitz` objects that share the same Core: one acts as generator and the other as discriminator.
3. **Download real samples** from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset using `datasets.load_dataset`:
   ```python
   from datasets import load_dataset
   ds = load_dataset('mnist', split='train')
   real_values = [x['image'].reshape(-1).numpy() / 255.0 for x in ds]
   ```
4. **Construct an `AdversarialLearner`** and call `learner.train(real_values)` to alternate generator and discriminator updates.
5. **Sample new data** after training by passing random noise vectors to the generator's `dynamic_wander` method.

**Complete Example**
```python
# project9_adversarial_learning.py
from config_loader import load_config
from marble_main import MARBLE
from adversarial_learning import AdversarialLearner

cfg = load_config()
generator = MARBLE(cfg['core']).neuronenblitz
discriminator = MARBLE(cfg['core']).neuronenblitz
learner = AdversarialLearner(generator, discriminator)
from datasets import load_dataset
ds = load_dataset('mnist', split='train')
real_values = [x['image'].reshape(-1).numpy() / 255.0 for x in ds]
learner.train(real_values)
noise = sample_noise(len(real_values[0][0]))
print(generator.dynamic_wander(noise))
```
Run this script to see generator and discriminator training in action.

## Project 10 – Autoencoder Learning (Frontier)

**Goal:** Reconstruct noisy inputs using an autoencoder built with Neuronenblitz.

1. **Enable the autoencoder module** by setting `autoencoder_learning.enabled: true` in `config.yaml` and choose values for `epochs`, `batch_size`, `noise_std` and `noise_decay`. The `noise_std` parameter sets the initial noise level while `noise_decay` reduces it after each epoch.
2. **Instantiate the classes**:
   ```python
   from autoencoder_learning import AutoencoderLearner
   auto = AutoencoderLearner(core, neuronenblitz)
   ```
3. **Download a real dataset**. The `load_digits` function from `sklearn.datasets` fetches handwritten digits as 8×8 images:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   values = [img / 16.0 for img in digits.data]
   ```
4. **Train** using `auto.train(values)` and inspect `auto.history` to see reconstruction losses decreasing over epochs.

**Complete Example**
```python
# project10_autoencoder.py
from config_loader import load_config
from marble_main import MARBLE
from autoencoder_learning import AutoencoderLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
auto = AutoencoderLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
values = [img / 16.0 for img in digits.data]
auto.train(values)
print(auto.history[-1])
```
Launch with `python project10_autoencoder.py` after enabling the module.

## Project 11 – Semi-Supervised Learning (Frontier)

**Goal:** Combine labeled and unlabeled data using the `SemiSupervisedLearner`.

1. **Enable the module** by editing `config.yaml` and setting `semi_supervised_learning.enabled: true`. Configure `epochs`, `batch_size` and the weight `unlabeled_weight` applied to the unsupervised loss.
2. **Create the learner** with your existing network:
   ```python
   from semi_supervised_learning import SemiSupervisedLearner
   learner = SemiSupervisedLearner(core, neuronenblitz)
   ```
3. **Download the digits dataset** and split it so only a fraction retains labels:
   ```python
   from sklearn.datasets import load_digits
   from sklearn.model_selection import train_test_split
   digits = load_digits()
   X_train, X_unlabeled, y_train, _ = train_test_split(
       digits.data, digits.target, test_size=0.8, random_state=42)
   labeled = list(zip(X_train, y_train))
   unlabeled = list(X_unlabeled)
   ```
4. **Train** using these lists with `learner.train(labeled, unlabeled)` and inspect `learner.history` to gauge progress.

**Complete Example**
```python
# project11_semi_supervised.py
from config_loader import load_config
from marble_main import MARBLE
from semi_supervised_learning import SemiSupervisedLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = SemiSupervisedLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X_train, X_unlabeled, y_train, _ = train_test_split(
    digits.data, digits.target, test_size=0.8, random_state=42)
labeled = list(zip(X_train, y_train))
unlabeled = list(X_unlabeled)
learner.train(labeled, unlabeled)
print(learner.history[-1])
```
Execute `python project11_semi_supervised.py` once the module is enabled.

## Project 12 – Federated Learning (Frontier)

**Goal:** Train multiple Neuronenblitz networks on separate datasets and combine them using federated averaging.**

1. **Enable federated mode** by setting `federated_learning.enabled: true` in `config.yaml`. Define the number of communication `rounds` and how many `local_epochs` each client trains before averaging.
2. **Instantiate clients**: create a separate `Core` and `Neuronenblitz` for each participant and pass these to a `FederatedAveragingTrainer` instance.
3. **Download a real dataset** to distribute among the clients. The MNIST digits can be fetched once and partitioned equally:
   ```python
   from datasets import load_dataset
   mnist = load_dataset('mnist', split='train')
   data_parts = np.array_split(list(mnist), len(clients))
   datasets = [[(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in part]
               for part in data_parts]
   ```
4. **Train round by round** by passing each client's portion to `trainer.train_round(client_data)` and check synchronisation by comparing synapse weights after training.

**Complete Example**
```python
# project12_federated.py
from config_loader import load_config
from federated_learning import FederatedAveragingTrainer
from marble_main import MARBLE

cfg = load_config()
clients = [MARBLE(cfg['core']) for _ in range(3)]
trainer = FederatedAveragingTrainer([c.neuronenblitz for c in clients])
from datasets import load_dataset
import numpy as np
mnist = load_dataset('mnist', split='train')
data_parts = np.array_split(list(mnist), len(clients))
datasets = [[(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in part]
            for part in data_parts]
for round_data in datasets:
    trainer.train_round(round_data)
```
This script launches a simple three‑client federated session.

## Project 13 – Curriculum Learning (Frontier)

**Goal:** Gradually introduce harder examples to improve stability.**

1. **Turn on curriculum learning** by setting `curriculum_learning.enabled: true` in `config.yaml` and provide a list under `schedule` describing when to introduce more difficult samples.
2. **Create the learner** using your network:
   ```python
   from curriculum_learning import CurriculumLearner
   learner = CurriculumLearner(core, neuronenblitz)
   ```
3. **Download a dataset** such as the handwritten digits and rank samples by the number of activated pixels to approximate difficulty:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   difficulty = digits.data.sum(axis=1)
   sorted_idx = difficulty.argsort()
   sorted_samples = [(digits.data[i], digits.target[i]) for i in sorted_idx]
   ```
4. **Call** `learner.train(sorted_samples)` to progressively feed in harder examples and track progress via `learner.history`.

**Complete Example**
```python
# project13_curriculum.py
from config_loader import load_config
from marble_main import MARBLE
from curriculum_learning import CurriculumLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = CurriculumLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
difficulty = digits.data.sum(axis=1)
sorted_idx = difficulty.argsort()
sorted_samples = [(digits.data[i], digits.target[i]) for i in sorted_idx]
learner.train(sorted_samples)
print(learner.history[-1])
```
Run `python project13_curriculum.py` to try curriculum learning.

## Project 14 – Meta Learning (Frontier)

**Goal:** Adapt quickly to new tasks using the Reptile algorithm.**

1. **Activate meta learning** in `config.yaml` by setting `meta_learning.enabled: true` and specify `epochs`, `inner_steps` and a meta learning rate `meta_lr`.
2. **Prepare tasks** using real data. Split the digits dataset into per-class tasks so each contains only samples from one digit:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   tasks = []
   for digit in range(10):
       mask = digits.target == digit
       pairs = list(zip(digits.data[mask], digits.target[mask]))
       tasks.append(pairs)
   ```
3. **Create the meta learner**:
   ```python
   from meta_learning import MetaLearner
   learner = MetaLearner(core, neuronenblitz)
   ```
4. **Iterate over epochs** calling `learner.train_step(tasks)` each time to perform the Reptile update.
5. **Review meta-loss** from `learner.history` to see how quickly the model adapts across tasks.

**Complete Example**
```python
# project14_meta_learning.py
from config_loader import load_config
from marble_main import MARBLE
from meta_learning import MetaLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = MetaLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
tasks = []
for digit in range(10):
    mask = digits.target == digit
    pairs = list(zip(digits.data[mask], digits.target[mask]))
    tasks.append(pairs)
for _ in range(cfg['meta_learning']['epochs']):
    learner.train_step(tasks)
print(learner.history[-1])
```
Launch the script to practice Reptile-style meta learning.

## Project 15 – Transfer Learning (Frontier)

**Goal:** Fine-tune a pretrained MARBLE model on a new dataset while freezing a subset of synapses.**

1. **Enable transfer learning** by setting `transfer_learning.enabled: true` in `config.yaml`. Choose `epochs`, `batch_size` and a `freeze_fraction` specifying what portion of synapses remain unchanged.
2. **Create the transfer learner** after loading an existing model (or training a base model) and passing the `Core` and `Neuronenblitz` objects to `TransferLearner`.
3. **Download a new dataset** such as Fashion‑MNIST and prepare `(input, target)` tuples:
   ```python
   from datasets import load_dataset
   ds = load_dataset('fashion_mnist', split='train')
   new_pairs = [(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in ds]
   ```
4. **Fine-tune** on these pairs by calling `learner.train(new_pairs)`.
5. **Tune `freeze_fraction`** to control how many synapses stay fixed during fine‑tuning and monitor `learner.history` to check performance on the new task.

**Complete Example**
```python
# project15_transfer.py
from config_loader import load_config
from marble_main import MARBLE
from transfer_learning import TransferLearner

cfg = load_config()
base = MARBLE(cfg['core'])
learner = TransferLearner(base.core, base.neuronenblitz)
from datasets import load_dataset
ds = load_dataset('fashion_mnist', split='train')
new_pairs = [(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in ds]
learner.train(new_pairs)
print(learner.history[-1])
```
Run `python project15_transfer.py` after enabling transfer learning.

## Project 16 – Continual Learning (Frontier)

**Goal:** Train sequential tasks while replaying previous examples.**

1. **Enable continual learning** in the configuration by setting `continual_learning.enabled: true` and provide values for `epochs` and the replay `memory_size`.
2. **Create the learner**:
   ```python
   from continual_learning import ReplayContinualLearner
   learner = ReplayContinualLearner(core, neuronenblitz)
   ```
3. **Download a sequence of datasets** such as Digits, Iris and Wine from `sklearn.datasets`:
   ```python
   from sklearn.datasets import load_digits, load_iris, load_wine
   datasets = [
       list(zip(load_digits().data, load_digits().target)),
       list(zip(load_iris().data, load_iris().target)),
       list(zip(load_wine().data, load_wine().target))
   ]
   ```
4. **Train sequentially** by calling `learner.train(data)` for each dataset and monitor reconstruction loss via `learner.history` to see how well the model retains previous knowledge.

**Complete Example**
```python
# project16_continual.py
from config_loader import load_config
from marble_main import MARBLE
from continual_learning import ReplayContinualLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = ReplayContinualLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits, load_iris, load_wine
datasets = [
    list(zip(load_digits().data, load_digits().target)),
    list(zip(load_iris().data, load_iris().target)),
    list(zip(load_wine().data, load_wine().target))
]
for data in datasets:
    learner.train(data)
print(learner.history[-1])
```
Execute the file to reproduce continual learning across tasks.

## Project 17 – Imitation Learning (Exploration)

**Goal:** Learn a policy directly from demonstration pairs.**

1. **Enable imitation mode** by setting `imitation_learning.enabled: true` in the configuration and choose `epochs` along with the `max_history` size that limits how many demonstrations are stored.
2. **Create the learner**:
   ```python
   from imitation_learning import ImitationLearner
   imitator = ImitationLearner(core, neuronenblitz)
   ```
3. **Download demonstration data** using the Hugging Face `datasets` library:
   ```python
   from datasets import load_dataset
   demos = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
   ```
4. **Record demonstrations** from a real environment such as `CartPole-v1` using `gymnasium`:
   ```python
   import gymnasium as gym
   env = gym.make('CartPole-v1')
   for _ in range(50):
       obs, _ = env.reset()
       done = False
       while not done:
           action = env.action_space.sample()
           imitator.record(obs, action)
           obs, _, done, _, _ = env.step(action)
   ```
   After recording, call `imitator.train()` (or `neuronenblitz.imitation_train()`) to learn from the history.
5. **Evaluate** the cloned policy by passing new inputs to `dynamic_wander` and observing the predicted actions.

**Complete Example**
```python
# project17_imitation.py
from config_loader import load_config
from marble_main import MARBLE
from imitation_learning import ImitationLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
imitator = ImitationLearner(marble.core, marble.neuronenblitz)
from datasets import load_dataset
demos = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
for step in demos["train"]:
    imitator.record(step["obs"], step["action"])
import gymnasium as gym
env = gym.make('CartPole-v1')
for _ in range(50):
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        imitator.record(obs, action)
        obs, _, done, _, _ = env.step(action)
imitator.train()
print(marble.brain.dynamic_wander(new_input))
```
Run `python project17_imitation.py` to train from demonstrations.

## Project 18 – Harmonic Resonance Learning (Novel)

**Goal:** Explore the experimental frequency-based paradigm.**

1. **Enable harmonic resonance** by editing the `harmonic_resonance_learning` section of `config.yaml` and setting `enabled: true` with parameters `epochs`, `base_frequency` and `decay`.
2. **Instantiate the learner**:
   ```python
   from harmonic_resonance_learning import HarmonicResonanceLearner
   learner = HarmonicResonanceLearner(core, neuronenblitz)
   ```
3. **Download a real time series** such as the [Jena climate dataset](https://github.com/philipperemy/keras-tutorials/blob/master/resources/jena_climate_2009_2016.csv.zip) programmatically and load it with `pandas`:
   ```python
   import urllib.request, zipfile, io, pandas as pd

   url = "https://github.com/philipperemy/keras-tutorials/raw/master/resources/jena_climate_2009_2016.csv.zip"
   with urllib.request.urlopen(url) as resp:
       with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
           zf.extractall()
   data = pd.read_csv('jena_climate_2009_2016.csv')
   values = list(zip(data['T (degC)'].values, data['p (mbar)'].values))
   ```
4. **Train** by repeatedly calling `learner.train_step(value, target)` for the specified number of epochs and observe frequency error in `learner.history` to understand how phase alignment evolves.

**Complete Example**
```python
# project18_harmonic.py
from config_loader import load_config
from marble_main import MARBLE
from harmonic_resonance_learning import HarmonicResonanceLearner
import urllib.request, zipfile, io, pandas as pd

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = HarmonicResonanceLearner(marble.core, marble.neuronenblitz)
url = "https://github.com/philipperemy/keras-tutorials/raw/master/resources/jena_climate_2009_2016.csv.zip"
with urllib.request.urlopen(url) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        zf.extractall()
data = pd.read_csv('jena_climate_2009_2016.csv')
values = list(zip(data['T (degC)'].values, data['p (mbar)'].values))
for value, target in values[:1000]:
    learner.train_step(value, target)
print(learner.history[-1])
```
Run `python project18_harmonic.py` to explore harmonic resonance learning.

## Project 19 – Synaptic Echo Learning (Novel)

**Goal:** Experiment with echo-modulated weight updates.**

1. **Enable synaptic echo** by setting `synaptic_echo_learning.enabled: true` in the YAML configuration and choose values for `epochs` and `echo_influence`.
2. **Instantiate the learner** ensuring the underlying Neuronenblitz has `use_echo_modulation=True`:
   ```python
   from synaptic_echo_learning import SynapticEchoLearner
   learner = SynapticEchoLearner(core, neuronenblitz)
   ```
3. **Download data** with clear sequential patterns such as the digits dataset and repeatedly call `learner.train_step(value, target)`:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   values = list(zip(digits.data, digits.target))
   ```
4. **Train** by iterating over `values` and calling `learner.train_step(v, t)` while monitoring `learner.history` and the synapse echo buffers to see how past activations influence current learning.

**Complete Example**
```python
# project19_synaptic_echo.py
from config_loader import load_config
from marble_main import MARBLE
from synaptic_echo_learning import SynapticEchoLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = SynapticEchoLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
values = list(zip(digits.data, digits.target))
for value, target in values:
    learner.train_step(value, target)
print(learner.history[-1])
```
Run `python project19_synaptic_echo.py` with echo modulation enabled.

## Project 20 – Fractal Dimension Learning (Novel)

**Goal:** Let MARBLE expand its representations when activity becomes complex.**

1. **Enable fractal dimension learning** by setting `fractal_dimension_learning.enabled: true` in the configuration and choose `epochs` along with the desired `target_dimension`.
2. **Instantiate the learner**:
   ```python
   from fractal_dimension_learning import FractalDimensionLearner
   learner = FractalDimensionLearner(core, neuronenblitz)
   ```
3. **Download real training pairs** from the digits dataset:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   pairs = list(zip(digits.data, digits.target))
   ```
4. **Train** with `learner.train(pairs)` and watch representation size via `core.rep_size` or `learner.history` to see when new dimensions are added.

**Complete Example**
```python
# project20_fractal.py
from config_loader import load_config
from marble_main import MARBLE
from fractal_dimension_learning import FractalDimensionLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = FractalDimensionLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
pairs = list(zip(digits.data, digits.target))
learner.train(pairs)
print(marble.core.rep_size)
```
Run `python project20_fractal.py` to see representations grow over time.

## Project 21 – Quantum Flux Learning (Novel)

**Goal:** Explore phase-modulated weight updates.**

1. **Enable quantum flux learning** in the configuration by setting `quantum_flux_learning.enabled: true` and choose values for `epochs` and the update `phase_rate`.
2. **Create the learner**:
   ```python
   from quantum_flux_learning import QuantumFluxLearner
   learner = QuantumFluxLearner(core, neuronenblitz)
   ```
3. **Download a dataset** such as the digits and iterate over the pairs:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   examples = list(zip(digits.data, digits.target))
   ```
4. **Train** by repeatedly calling `learner.train_step(inp, tgt)` for each pair and track phases in `learner.phases` to understand how the system evolves over time.

**Complete Example**
```python
# project21_quantum_flux.py
from config_loader import load_config
from marble_main import MARBLE
from quantum_flux_learning import QuantumFluxLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = QuantumFluxLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
examples = list(zip(digits.data, digits.target))
for inp, tgt in examples:
    learner.train_step(inp, tgt)
print(learner.phases[-1])
```
Run `python project21_quantum_flux.py` to experiment with quantum flux updates.

## Project 22 – Dream Reinforcement Synergy (Novel)

**Goal:** Combine dreaming with reinforcement-like updates.**

1. **Enable dream reinforcement** by setting `dream_reinforcement_learning.enabled: true` in the YAML file and configure `episodes`, `dream_cycles` and `dream_strength`.
2. **Instantiate the learner**:
   ```python
   from dream_reinforcement_learning import DreamReinforcementLearner
   learner = DreamReinforcementLearner(core, neuronenblitz)
   ```
3. **Download a demonstration dataset** using the Hugging Face `datasets` library:
   ```python
   from datasets import load_dataset
   dream_demo = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
   ```
4. **Use a real environment** such as `CartPole-v1` to generate `(input, target)` pairs:
   ```python
   import gymnasium as gym
   env = gym.make('CartPole-v1')
   def sample_episode():
       obs, _ = env.reset()
       done = False
       while not done:
           action = env.action_space.sample()
           next_obs, reward, done, _, _ = env.step(action)
           yield obs, reward
           obs = next_obs
   ```
5. **Train episodes** by repeatedly calling `learner.train_episode(inp, tgt)` for each step. Imaginary updates occur after each real step; `dream_cycles` controls how many of these dreaming iterations happen.

**Complete Example**
```python
# project22_dream_reinforcement.py
from config_loader import load_config
from marble_main import MARBLE
from dream_reinforcement_learning import DreamReinforcementLearner

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = DreamReinforcementLearner(marble.core, marble.neuronenblitz)
from datasets import load_dataset
dream_demo = load_dataset("deep-rl-datasets", "cartpole-expert-v1")
import gymnasium as gym
env = gym.make('CartPole-v1')
def sample_episode():
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _, _ = env.step(action)
        yield obs, reward
        obs = next_obs
for episode in range(cfg['dream_reinforcement_learning']['episodes']):
    for inp, tgt in sample_episode():
        learner.train_episode(inp, tgt)
```
Execute `python project22_dream_reinforcement.py` to see the synergy in action.

## Project 23 – Omni Learning Paradigm (Advanced)

**Goal:** Train using every supported paradigm at once.**

1. **Combine cores** by creating several `Core` objects and merging them with `interconnect_cores` from `core_interconnect`.
2. **Instantiate one Neuronenblitz** using the combined core so the learner sees a unified network.
3. **Create an `OmniLearner`** with the merged core and single Neuronenblitz instance.
4. **Download a dataset** such as MNIST and prepare training examples shared by all paradigms:
   ```python
   from datasets import load_dataset
   ds = load_dataset('mnist', split='train')
   examples = [(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in ds]
   ```
5. **Train** by providing these examples to `learner.train(examples, epochs=5)`. All paradigms run sequentially, leveraging interconnection synapses so multiple cores behave as one integrated system.

**Complete Example**
```python
# project23_omni.py
from config_loader import load_config
from marble_main import MARBLE
from core_interconnect import interconnect_cores
from omni_learning import OmniLearner

cfg = load_config()
cores = [MARBLE(cfg['core']).core for _ in range(3)]
combined = interconnect_cores(cores)
neuronenblitz = MARBLE(cfg['core']).neuronenblitz
learner = OmniLearner(combined, neuronenblitz)
from datasets import load_dataset
ds = load_dataset('mnist', split='train')
examples = [(x['image'].reshape(-1).numpy() / 255.0, x['label']) for x in ds]
learner.train(examples, epochs=5)
```
Run `python project23_omni.py` to test all paradigms together.

## Project 23b – Unified Multi-Paradigm Learning

**Goal:** Dynamically choose which paradigm to emphasize during training.**

1. **Enable unified learning** by setting `unified_learning.enabled: true` in
   `config.yaml`.
2. **Create a `UnifiedLearner`** using an existing core, Neuronenblitz and a
   dictionary of sub-learners:
   ```python
   from unified_learning import UnifiedLearner
   learners = {
       'contrastive': ContrastiveLearner(core, nb),
       'hebbian': HebbianLearner(core, nb),
       'autoencoder': AutoencoderLearner(core, nb),
   }
   learner = UnifiedLearner(core, nb, learners)
   ```
3. **Train** normally using `learner.train_step((inp, tgt))` inside your loop.
   The gating network assigns a weight to each learner every step, modulating
   `neuronenblitz.plasticity_modulation`. Decisions are stored in the file set
   by `log_path`.

## Project 24 – Continuous Weight Field Learning (Experimental)

**Goal:** Train a smooth weight field that adapts to each input.**

1. **Enable the learner** by setting `continuous_weight_field_learning.enabled: true` in `config.yaml` and optionally tweak `num_basis` or `bandwidth`.
2. **Instantiate the learner**:
   ```python
   from continuous_weight_field_learning import ContinuousWeightFieldLearner
   learner = ContinuousWeightFieldLearner(core, neuronenblitz)
   ```
3. **Download a dataset**. The diabetes set from scikit-learn provides real regression targets:
   ```python
   from sklearn.datasets import load_diabetes
   ds = load_diabetes()
   samples = list(zip(ds.data[:, 0], ds.target))
   ```
4. **Train** using `learner.train(samples, epochs=2)` and monitor `learner.history` for the squared error over time.

**Complete Example**
```python
# project24_cwfl.py
from config_loader import load_config
from marble_main import MARBLE
from continuous_weight_field_learning import ContinuousWeightFieldLearner
from sklearn.datasets import load_diabetes

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = ContinuousWeightFieldLearner(marble.core, marble.neuronenblitz)
ds = load_diabetes()
samples = list(zip(ds.data[:, 0], ds.target))
learner.train(samples, epochs=2)
```
Run `python project24_cwfl.py` to see the field adapt across the dataset.

## Project 25 – Neural Schema Induction (Theory)

**Goal:** Demonstrate structural learning of repeated reasoning patterns.**

1. **Enable schema induction** by setting `neural_schema_induction.enabled: true`
   in `config.yaml` and adjust `support_threshold` if needed.
2. **Instantiate the learner**:
   ```python
   from neural_schema_induction import NeuralSchemaInductionLearner
   learner = NeuralSchemaInductionLearner(core, neuronenblitz)
   ```
3. **Download a dataset** such as the digits and create a list of inputs only:
   ```python
   from sklearn.datasets import load_digits
   digits = load_digits()
   inputs = [x.reshape(-1).astype(float) for x in digits.data]
   ```
4. **Train** using `learner.train(inputs, epochs=2)` and inspect
   `learner.schemas` to see the discovered patterns.

**Complete Example**
```python
# project25_nsi.py
from config_loader import load_config
from marble_main import MARBLE
from neural_schema_induction import NeuralSchemaInductionLearner
from sklearn.datasets import load_digits

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = NeuralSchemaInductionLearner(marble.core, marble.neuronenblitz)
digits = load_digits()
inputs = [x.reshape(-1).astype(float) for x in digits.data]
learner.train(inputs, epochs=2)
print(len(learner.schemas))
```
Run `python project25_nsi.py` to see schema neurons emerge.

## Project 26 – Conceptual Integration

**Goal:** Blend concepts structurally to spur creative abstractions.**

1. **Enable conceptual integration** by setting `conceptual_integration.enabled: true` in `config.yaml`.
2. **Instantiate the learner**:
   ```python
   from conceptual_integration import ConceptualIntegrationLearner
   learner = ConceptualIntegrationLearner(core, neuronenblitz,
                                          blend_probability=0.5,
                                          similarity_threshold=0.2)
   ```
3. **Download any regression dataset** such as Boston housing:
   ```python
   from sklearn.datasets import load_boston
   ds = load_boston()
   samples = list(ds.data[:, 0])
   ```
4. **Train** with `learner.train(samples, epochs=2)` and observe
   that `len(core.neurons)` grows as new concept nodes are inserted.

**Complete Example**
```python
# project26_cip.py
from config_loader import load_config
from marble_main import MARBLE
from conceptual_integration import ConceptualIntegrationLearner
from sklearn.datasets import load_boston

cfg = load_config()
marble = MARBLE(cfg['core'])
learner = ConceptualIntegrationLearner(marble.core, marble.neuronenblitz,
                                       blend_probability=0.5,
                                       similarity_threshold=0.2)
ds = load_boston()
samples = list(ds.data[:, 0])
learner.train(samples, epochs=2)
print(len(marble.core.neurons))
```
Run `python project26_cip.py` to watch concepts emerge through blending.

## Project 27 – Hybrid Memory Architecture (Advanced)

**Goal:** Demonstrate storing and recalling information with the new hybrid memory system.

1. **Prepare a configuration** enabling the hybrid memory:
   ```yaml
   hybrid_memory:
     vector_store_path: "vectors.pkl"
     symbolic_store_path: "symbols.pkl"
   ```
2. **Create the training script.** It saves a few values and then queries them:
   ```python
   # project27_hybrid_memory.py
   from config_loader import load_config, create_marble_from_config

   cfg = load_config()
   cfg["hybrid_memory"] = {
       "vector_store_path": "vectors.pkl",
       "symbolic_store_path": "symbols.pkl",
   }
   marble = create_marble_from_config(cfg)
   marble.hybrid_memory.store("a", 1.0)
   marble.hybrid_memory.store("b", 2.0)
   print(marble.hybrid_memory.retrieve(1.0, top_k=1))
   ```
3. **Run the script** with `python project27_hybrid_memory.py`. The retrieved list should contain the key `'a'` showing the memory system found the correct entry.

## Project 28 – Convenience Interface Functions (Easy)

**Goal:** Utilise the high-level helper functions in `marble_interface` for rapid experiments.

1. **Load a small dataset** directly from Hugging Face:
   ```python
   from marble_interface import load_hf_dataset

   train_pairs = load_hf_dataset(
       "mnist", "train[:100]", input_key="image", target_key="label"
   )
   ```
2. **Create and train a MARBLE system** using a pandas dataframe:
   ```python
   import pandas as pd
   from marble_interface import new_marble_system, train_from_dataframe

   df = pd.DataFrame({"input": [0.1, 0.2], "target": [0.2, 0.4]})
   marble = new_marble_system()
   train_from_dataframe(marble, df, epochs=1)
   ```
3. **Evaluate and serialise the core** for later use:
   ```python
   from marble_interface import evaluate_marble_system, export_core_to_json

   mse = evaluate_marble_system(marble, train_pairs[:10])
   json_core = export_core_to_json(marble)
   with open("core.json", "w", encoding="utf-8") as f:
       f.write(json_core)
   ```
4. **Restore the core** to a fresh system when needed:
   ```python
   from marble_interface import import_core_from_json

   with open("core.json", "r", encoding="utf-8") as f:
       loaded_json = f.read()
   restored = import_core_from_json(loaded_json)
   ```



## Project 29 – Streamlit Playground (Easy)

**Goal:** Interactively explore MARBLE through a web-based GUI.

1. **Install Streamlit** (already in `requirements.txt`).
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the playground** from the repository root:
   ```bash
   streamlit run streamlit_playground.py
   ```
3. **Provide a configuration.** Enter a YAML path, upload a YAML file or paste
   YAML directly in the sidebar. Press **Initialize MARBLE** to create the
   system.
4. **Upload a dataset** in CSV, JSON or ZIP format. ZIP archives may contain
   `dataset.csv`/`dataset.json` or paired `inputs/` and `targets/` folders with
   images or audio files. Any mixture of numbers, text, images and audio is
   supported. Select the desired number of epochs and click **Train**.
5. **Search Hugging Face datasets** using the sidebar field. Enter a query,
   click **Search Datasets** and choose a result to fill the dataset name
   automatically.
6. **Perform inference** by providing a numeric value, text, image or audio
   sample under the *Inference* section and pressing **Infer**. Training metrics
   appear automatically after each run.
7. **Save and load models** using the sidebar controls. You can also export or
   import the core JSON to share systems between sessions.
8. **Switch to Advanced mode** to access every function in
   ``marble_interface``. The playground displays each function's docstring and
   generates widgets for all parameters so you can call any operation directly.
9. **Build pipelines** on the *Pipeline* tab. Add steps from
   ``marble_interface`` or any repository module, then press **Run Pipeline** to
   execute them sequentially. This lets you combine training, evaluation and
   analysis commands without leaving the UI.
10. **View the core graph** on the *Visualization* tab. Press **Generate Graph**
   to see an interactive display of neurons and synapses.
11. **Edit configuration** from the *Config Editor* tab. Specify any dot-
    separated parameter path and new value to update the YAML in place. Press
    **Reinitialize** to rebuild the system with the modified settings.
12. **Consult the YAML manual** from the sidebar while adjusting parameters.
13. **Convert Hugging Face models** on the *Model Conversion* tab. Enter a
    model search query, preview the architecture and click **Convert to
    MARBLE** to initialize a system from the pretrained weights.
14. **Experiment with offloading** using the *Offloading* tab. Start a
    ``RemoteBrainServer`` or create a torrent client directly from the
    interface and attach it to the active MARBLE instance. Use the provided
    buttons to offload high‑attention lobes to the remote server or share them
    with peers via torrent.
15. **Monitor progress** on the *Metrics* tab. Loss, memory usage and other
    statistics are plotted live so you can observe training behaviour.
16. **Check system resources** on the *System Stats* tab. RAM and GPU usage are
    displayed so you can monitor consumption while experimenting.
17. **Tune neuromodulatory signals** on the *Neuromodulation* tab. Adjust the
    sliders for arousal, stress and reward or set an emotion string, then press
    **Update Signals** to modify MARBLE's internal context on the fly.
18. **Read the documentation** on the *Documentation* tab. Every markdown file
    in the repository, including the architecture overview, configurable
    parameters list and machine learning handbook, can be opened here for quick
    reference.
19. **Browse source code** on the *Source Browser* tab. Select any module and
    click **Show Source** to view its implementation without leaving the
    playground.
20. **Run unit tests** on the *Tests* tab. Select one or more test files and
    click **Run Tests** to verify everything works as expected.

Uploaded datasets are previewed directly in the sidebar so you can verify their
contents before training. The currently active YAML configuration is also shown
in an expandable panel for quick reference. Use these previews to ensure your
data and settings are correct before experimenting.

The playground provides toggles for dreaming and autograd features so you can
experiment with MARBLE's advanced capabilities without writing code.
