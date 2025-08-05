# MARBLE Step-by-Step Tutorial

This tutorial demonstrates every major component of MARBLE through a series of projects. Each project builds on the previous one and introduces new functionality. The instructions below walk through every step in detail so that you can replicate the experiments exactly.

## Prerequisites

1. **Install dependencies** so that all modules are available. Run the following command inside the repository root and wait for the installation to finish:
   ```bash
   pip install -r requirements.txt
   ```
   After the packages are installed you can verify the environment by running `python -c "import marble_main"` which should finish silently.
2. **Review the documentation**. Read [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for a high level view of MARBLE and consult [yaml-manual.txt](yaml-manual.txt) for an explanation of every configuration option.
3. **Open the interactive notebook** found in ``notebooks/tutorial.ipynb`` if
   you prefer running the tutorial step by step inside Jupyter.

4. **Use the command line interface**. The `cli.py` script allows training from
   the terminal without writing custom code. Scheduler and early-stopping
   parameters can be specified on the command line:
   ```bash
   python cli.py --config config.yaml --train path/to/data.csv --epochs 10 \
       --lr-scheduler cosine --scheduler-steps 20 --early-stopping-patience 5 \
       --save trained_marble.pkl
   ```
   Replace the dataset path with your own CSV or JSON file. The optional
   `--validate` flag specifies a validation dataset.

## Project: Generating a Workflow Template

This project demonstrates the template generator that creates starter
pipeline code.  The generator detects available hardware and embeds CPU
or GPU defaults into the produced script.

1. **List available templates** to see the patterns that can be
   generated:

   ```bash
   python workflow_template_generator.py --list
   ```

2. **Generate a classification template** and save it to a file:

   ```bash
   python workflow_template_generator.py classification demo_pipeline.py
   ```

   Use `--device cpu` or `--device cuda` to override the automatic
   hardware detection.

3. **Run the generated pipeline**.  It contains a synthetic training
   loop that executes end-to-end:

   ```bash
   python demo_pipeline.py
   ```

### Troubleshooting

- ``ValueError: Unknown template`` – ensure the template name matches one
  listed by the ``--list`` command.
- ``RuntimeError: CUDA error`` – if your system lacks a GPU, regenerate
  the template with ``--device cpu``.

## Project: Checkpointing and Resuming Pipelines

This project shows how to persist a pipeline with an explicit dataset version
and later resume it even on different hardware.

1. **Create a simple pipeline** and write it to JSON:

   ```python
   from highlevel_pipeline import HighLevelPipeline

   pipe = HighLevelPipeline()
   with open("demo.json", "w", encoding="utf-8") as f:
       f.write(pipe.to_json())
   ```

2. **Run the pipeline and save a checkpoint** using the CLI:

   ```bash
   python highlevel_pipeline_cli.py checkpoint demo.json demo.pkl \
       --config config.yaml --dataset-version v1 --device cpu
   ```

3. **Resume from the checkpoint**. The command can target CPU or GPU:

   ```bash
   python highlevel_pipeline_cli.py resume demo.pkl --config config.yaml \
       --device gpu
   ```

4. **Inspect the stored dataset version** to confirm it persisted:

   ```python
   from highlevel_pipeline import HighLevelPipeline

   loaded = HighLevelPipeline.load_checkpoint("demo.pkl")
   print(loaded.dataset_version)
   ```

### Data Loading and Tokenization

All examples below rely on the **new** :class:`DataLoader` and the
``dataset_loader.load_dataset`` utility. The loader transparently compresses
and serialises values and, when supplied with a tokenizer, converts text into
token IDs. The snippet shows how to create a ``DataLoader`` with a built in
WordPiece tokenizer and load a CSV file:

```python
from tokenizer_utils import built_in_tokenizer
from dataset_loader import load_dataset
from marble import DataLoader

# Use a built-in tokenizer when working with text
dataloader = DataLoader(tokenizer=built_in_tokenizer("bert_wordpiece"))
pairs = load_dataset("path/to/data.csv", dataloader=dataloader)
```

For purely numeric or image datasets simply use ``DataLoader()`` without a
tokenizer. When you need full control over the binary representation you can
wrap your ``(input, target)`` pairs in :class:`BitTensorDataset`. This converts
each object into a tensor of bits and optionally compresses repeated patterns
through a shared vocabulary.

### Streaming large CSV files

MARBLE can process massive datasets without loading them entirely into memory
using :class:`StreamingCSVLoader`. This iterator reads one line at a time and
records its byte offset in ``data.csv.meta.json`` so subsequent runs resume
where the previous one stopped.

```python
from dataset_loader import StreamingCSVLoader
from tokenizer_utils import built_in_tokenizer

tok = built_in_tokenizer("char_bpe")
tok.train_from_iterator(["hello", "world"], vocab_size=10)

for row in StreamingCSVLoader("data.csv", tokenizer=tok):
    print(row["input_ids"], row["target"])
```

If you set ``dataset.encryption_key`` in ``config.yaml`` the loader encrypts all
objects before writing them to disk and automatically decrypts them when
loading. Use the same key on every machine that processes the dataset to handle
them correctly.

Set ``dataloader.tokenizer_type: bert_wordpiece`` or ``tokenizer_json`` in
``config.yaml`` to use the same tokenizer when constructing ``MARBLE``. Each
project example assumes a ``dataloader`` prepared this way and passes it to
``load_dataset``.

### Project: Streaming Dataset Pipeline

This project demonstrates how to process a streaming dataset through a pipeline.

1. **Download a dataset** from the Hugging Face Hub using its streaming API:

   ```python
   from datasets import load_dataset

   raw = load_dataset("ag_news", split="train", streaming=True)
   ```

2. **Encode records into a BitTensorDataset** so that the pipeline receives
   tensors:

   ```python
   from tokenizer_utils import built_in_tokenizer
   from marble import DataLoader
   from bit_tensor_dataset import BitTensorDataset

   tok = built_in_tokenizer("bert_wordpiece")
   dataloader = DataLoader(tokenizer=tok)
   pairs = []
   for rec in raw.take(200):
       inp = dataloader.encode(rec["text"])
       tgt = dataloader.encode(str(rec["label"]))
       pairs.append((inp, tgt))
   dataset = BitTensorDataset(pairs, device="cpu")
   ```

3. **Create a pipeline with the streaming step and run it**:

   ```python
   from pipeline import Pipeline
   import marble_interface

   pipe = Pipeline([
       {
           "func": "streaming_dataset_step",
           "module": "marble_interface",
           "params": {"dataset": dataset, "batch_size": 16, "prefetch": 4, "device": "cuda"},
       }
   ])
   batches = pipe.execute()[0]
   print("First batch shape:", batches[0]["inputs"].shape)
   ```

The background producer moves tensors to the target device and the pipeline
automatically consumes the stream, yielding a list of batches for further
processing.

### Customising Steps with Hooks

The :class:`pipeline.Pipeline` supports registering callables that run before
and after individual steps. Hooks receive the step specification, an optional
MARBLE instance and the ``torch.device`` in use. Post hooks also receive the
step's result and may replace it. Hooks execute in the order they were
registered, allowing deterministic composition.

```python
import torch
from pipeline import Pipeline

def double(x: torch.Tensor) -> torch.Tensor:
    return x * 2

pipe = Pipeline()
pipe.add_step("double", module="__main__", params={"x": torch.tensor([1., 2.])}, name="d")

def pre_scale(step, marble, device):
    step["params"]["x"] = step["params"]["x"].to(device) * 3

def post_add(step, result, marble, device):
    return result + 1

pipe.register_pre_hook("d", pre_scale)
pipe.register_post_hook("d", post_add)
print(pipe.execute()[0])  # tensor([7., 13.])
```

Avoid side effects inside hooks and promptly release any GPU tensors to prevent
memory leaks.

### Interactive Step Debugging

During experimentation it can be helpful to pause after each pipeline step and
inspect what was passed in and what came out. ``Pipeline.enable_interactive_debugging``
registers hooks that capture this information and, by default, drop you into
``pdb`` before and after every step so you can explore the state interactively.

```python
import torch
from pipeline import Pipeline

def add_value(tensor: torch.Tensor, value: float, device: str) -> torch.Tensor:
    return tensor.to(device) + value

pipe = Pipeline()
pipe.add_step(
    "add_value",
    module="__main__",
    params={"tensor": torch.ones(2), "value": 5},
)
debugger = pipe.enable_interactive_debugging(interactive=False)  # set True for pdb
pipe.execute()
print(debugger.inputs[pipe.steps[0]["name"]])
print(debugger.outputs[pipe.steps[0]["name"]])
```

### Visualising Pipelines as Graphs

Pipelines can be rendered as graphs for inspection or documentation. The
``pipeline_to_networkx`` helper converts a sequence of step dictionaries into a
directed ``networkx`` graph that expands macros and branches automatically.

```python
from networkx_interop import pipeline_to_networkx

g = pipeline_to_networkx(pipe.steps)
```

The resulting graph can be shown with any ``networkx`` layout or converted to a
MARBLE ``Core`` using ``pipeline_to_core`` for visualisation with the MARBLE
graph builder.

### Caching Step Results

When experimenting it is useful to skip recomputation. Supply a directory via
``cache_dir`` to ``Pipeline.execute`` to persist each step's output. On later
runs, unchanged steps load their cached result directly from disk and tensors
are automatically restored to the active CPU or GPU device.

```python
pipe.execute(cache_dir="cache")  # first run stores results
pipe.execute(cache_dir="cache")  # second run loads from disk
```

## Project 1 – Numeric Regression (Easy)

**Goal:** Train MARBLE on a simple numeric dataset.

1. **Download the data programmatically** so you have a local copy of the wine quality dataset:
   ```python
   import urllib.request

   url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
   urllib.request.urlretrieve(url, "winequality-red.csv")
   ```
   After the download completes you should see `winequality-red.csv` in the directory.
2. **Generate a quick synthetic dataset** using the helper in `synthetic_dataset.py` if you just want to experiment with the training loop:
   ```python
   from synthetic_dataset import generate_sine_wave_dataset

   train_examples = generate_sine_wave_dataset(200, noise_std=0.05, seed=0)
   ```
   This produces `(input, target)` pairs following a noisy sine wave.
3. **Run the dataset through a pipeline for automatic training.** Any step
   whose name contains `dataset` triggers a training loop for an attached
   `Neuronenblitz` instance:
   ```python
   from pipeline import Pipeline
   from marble_core import Core
   from marble_neuronenblitz import Neuronenblitz
   from tests.test_core_functions import minimal_params

   core = Core(minimal_params())
   nb = Neuronenblitz(core)
   pipe = Pipeline([
       {"module": "synthetic_dataset", "func": "generate_sine_wave_dataset"}
   ])
   pipe.execute(marble=nb)
   print(f"trained on {len(nb.get_training_history())} examples")
   ```
   The pipeline detects the dataset step and inserts a training loop
   automatically, moving tensors to GPU when available.
4. **Prepare the dataset** using the unified loader so objects are encoded
   consistently:
   ```python
   from dataset_loader import load_dataset

   pairs = load_dataset('winequality-red.csv', dataloader=dataloader)
   ```
   `pairs` is a list of `(input, target)` tuples ready for training.
   ```python
   from sklearn.model_selection import train_test_split
   train_examples, val_examples = train_test_split(pairs, test_size=0.1, random_state=42)
   ```
5. **Edit configuration**. Open `config.yaml` and modify the values under `core` to adjust the representation size and other parameters. Save the file after your edits.
   To experiment with sparser communication set `attention_dropout` to a value between `0.0` and `1.0`. Higher values randomly ignore more incoming messages during attention-based updates.
   You can also introduce oscillatory gating of synaptic weights by setting `global_phase_rate` to a value above `0.0`. Each call to `run_message_passing` then advances the internal `global_phase`, modulating every synapse via a cosine of its individual `phase`.
   You may also provide a *partial* YAML file containing only the settings you
   wish to override. `load_config` merges it with the defaults from
   `config.yaml` automatically.
6. **Create a MARBLE instance** from the configuration:
   ```python
   from marble_main import MARBLE
   from config_loader import load_config

   cfg = load_config()
   marble = MARBLE(cfg['core'])
   ```
7. **Train the model** on the prepared data:
   ```python
   marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
   ```
   Training progress is visualised with a sidebar progress bar in the Streamlit GUI.
8. **Monitor progress** with `MetricsVisualizer` which plots loss and memory usage. Adjust the `fig_width` and `color_scheme` options under `metrics_visualizer` in `config.yaml` to change the appearance.
   The pipeline core also broadcasts ``pipeline_progress`` events containing
   ``step``, ``index``, ``total``, ``device`` and ``status`` fields. The
   Streamlit GUI subscribes to these events and renders live updates—a progress
   bar on desktop layouts and textual percentages on mobile. If no updates
   appear, ensure JavaScript is enabled and the page URL includes the
   ``device`` query parameter.  These progress events can also be sent to a
   remote experiment tracker by calling ``attach_tracker_to_events`` with a
   tracker instance.  Each pipeline step is validated against a JSON schema when
   added or loaded so configuration errors are surfaced early.
9. **Export the trained model automatically** by providing `export_path` when executing a pipeline. The final step serializes the core to disk on the currently active device (CPU or GPU):
   ```python
   from pipeline import Pipeline
   from marble_core import Core
   from marble_neuronenblitz import Neuronenblitz
   from tests.test_core_functions import minimal_params

   core = Core(minimal_params())
   nb = Neuronenblitz(core)
   pipe = Pipeline()
   pipe.execute(marble=nb, export_path="core.json")
   ```
   Use `export_format="onnx"` to emit an ONNX graph instead of JSON.
10. **View metrics in your browser** by enabling `metrics_dashboard.enabled`. Set `window_size` to control the moving-average smoothing of the curves.
11. **Gradually reduce regularization** by setting `dropout_probability` and `dropout_decay_rate` under `neuronenblitz`. A decay rate below `1.0` multiplies the current dropout value after each epoch.
12. **Search hyperparameters** directly through the pipeline using `Pipeline.hyperparameter_search`:
   ```python
   from pipeline import Pipeline

   pipe = Pipeline()
   pipe.add_step('train_step', params={'epochs': 3}, name='train')
   def score(outputs):
       return outputs[-1]  # assume last step returns validation loss

   grid = {'train.epochs': [1, 2]}
   results = pipe.hyperparameter_search(grid, score, marble=marble)
   print('Best params:', results[0])
   ```
   The helper moves tensors back to the CPU for scoring and empties the GPU cache between trials so even long searches run reliably on limited hardware.

**Complete Example**
```python
# project1_numeric_regression.py
import urllib.request
from sklearn.model_selection import train_test_split
from marble_main import MARBLE
from config_loader import load_config
from marble import DataLoader
from dataset_loader import load_dataset

pairs_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
urllib.request.urlretrieve(pairs_url, "winequality-red.csv")
dataloader = DataLoader()  # numeric dataset, tokenizer not required
pairs = load_dataset("winequality-red.csv", dataloader=dataloader)
train_examples, val_examples = train_test_split(pairs, test_size=0.1, random_state=42)

cfg = load_config()
marble = MARBLE(cfg['core'])
marble.brain.train(train_examples, epochs=10, validation_examples=val_examples)
```
Run this script with `python project1_numeric_regression.py` to reproduce the first project end-to-end.

### Sharing datasets across machines

For multi-node experiments, start ``DatasetCacheServer`` on one machine and set
``dataset.cache_url`` in ``config.yaml`` so that other nodes download files from
the cache before accessing the internet. This dramatically reduces duplicated
traffic and keeps datasets consistent across workers.

This project introduces the **Core**, **Neuronenblitz** and **Brain** objects along with the data compression pipeline.

All following project scripts assume a ``dataloader`` created as in
Project&nbsp;1 so that text input is tokenized consistently. Only the dataset
URL or loading routine changes.

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
5. **Tune caching** using the ``wander_cache_ttl`` parameter in ``config.yaml`` to control how long ``dynamic_wander`` results remain valid. Increasing the value reuses paths more aggressively while ``0`` disables expiry.
6. **Speed up wandering** by enabling ``subpath_cache_size`` and ``subpath_cache_ttl`` under ``neuronenblitz``. This stores frequently used path prefixes so subsequent runs can recombine them without recomputing every step.
7. **Average parallel wanderers** by setting ``parallel_wanderers`` above ``1`` and ``parallel_update_strategy: average`` in ``config.yaml``. Each wanderer explores independently and their weight updates are averaged, producing smoother training dynamics.
8. **Bias exploration** toward informative routes by adjusting ``gradient_path_score_scale``. Set ``use_gradient_path_scoring: true`` to factor gradient magnitude into path selection. Enabling ``rms_gradient_path_scoring`` switches the metric from the sum of absolute last gradients to the root-mean-square of RMSProp statistics, favoring paths that consistently produce strong updates.
9. **Experiment with evolutionary functions** to mutate or prune synapses:
   ```python
   mutated, pruned = marble.brain.evolve(mutation_rate=0.02, prune_threshold=0.05)
   ```
   Mutations add noise to synapses while pruning removes the least useful ones.
10. **Enable dreaming** by setting `dream_enabled: true` in `config.yaml`. Parameters like `dream_num_cycles`, `dream_interval`, and the new `dream_replay_buffer_size` and `dream_replay_batch_size` control how frequently dream cycles run and how many past experiences they consolidate.

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

raw_pairs = load_cifar_batches('cifar-10-batches-py')
cfg = load_config()
dataloader = DataLoader()  # image data does not need tokenization
train_examples = [(dataloader.encode(x), dataloader.encode(y)) for x, y in raw_pairs]
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
    client = RemoteBrainClient('http://remote_host:8000', max_retries=5,
                               backoff_factor=1.0)
   marble = MARBLE(cfg['core'], remote_client=client)
   ```
   The client automatically retries transient failures using an exponential
   backoff (0.5, 1.0, 2.0 … seconds by default). Adjust ``max_retries`` and
   ``backoff_factor`` here or via the CLI flags ``--remote-retries`` and
   ``--remote-backoff``. If all retries fail the original error is raised so
   fatal issues are not hidden.
3. **Download a dataset** such as digits using `sklearn.datasets.load_digits()` for offloaded training:
   ```python
   from sklearn.datasets import load_digits
   from marble import DataLoader
   dataloader = DataLoader()
   digits = load_digits()
   train_pairs = [
       (dataloader.encode(x), dataloader.encode(y))
       for x, y in zip(digits.data, digits.target)
   ]
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
from marble import DataLoader

# Run this on the remote machine
server = RemoteBrainServer(port=8000)
server.start()

# On the training machine
cfg = load_config()
client = RemoteBrainClient('http://remote_host:8000')
marble = MARBLE(cfg['core'], remote_client=client)
from sklearn.datasets import load_digits
digits = load_digits()
dataloader = DataLoader()  # digits are numeric, no tokenizer
train_pairs = [(dataloader.encode(x), dataloader.encode(y)) for x, y in zip(digits.data, digits.target)]
marble.brain.offload_enabled = True
marble.brain.offload_high_attention(threshold=0.5)
```
Execute the file on each machine as indicated to experiment with remote offloading.

Remote offloading demonstrates **RemoteBrainServer**, **RemoteBrainClient** and the optional torrent‑based distribution.

### Custom remote hardware tiers

For specialised accelerators, implement a remote hardware plugin exposing a
``get_remote_tier`` factory. Set its import path under
``remote_hardware.tier_plugin`` in ``config.yaml``. During training the core
will delegate heavy computations to the custom tier, enabling seamless use of
non‑standard devices. Refer to [public_api.md](docs/public_api.md#remote-hardware-plugins)
for the programming interface. Individual pipeline steps can opt in to remote
execution by including a ``tier`` field with the name of the registered tier.
Steps without this field continue to run locally on CPU or GPU as usual.

## Project 3b – Remote Inference API (Medium)

**Goal:** Serve a trained MARBLE brain over HTTP for lightweight inference.

1. **Create a brain** and start the API:
   ```python
   from tests.test_core_functions import minimal_params
   from marble_core import Core, DataLoader
   from marble_neuronenblitz import Neuronenblitz
   from marble_brain import Brain
   from web_api import InferenceServer

   core = Core(minimal_params())
   nb = Neuronenblitz(core)
    brain = Brain(core, nb, DataLoader())  # numeric inference values
   server = InferenceServer(brain)
   server.start()
   ```
2. **Send requests** to `http://localhost:5000/infer` with JSON
   `{"input": 0.42}` and read back the numeric output.
3. **Stop the server** by calling `server.stop()` when finished.

This project highlights how MARBLE can integrate with external services through
a minimal web API.

## Project 3c – Config Synchronisation (Medium)

**Goal:** Keep configuration files consistent across multiple machines.

1. **Start the watcher** on the main node:
   ```python
   from config_sync_service import ConfigSyncService
   svc = ConfigSyncService('config.yaml', ['/mnt/nodeA/cfg.yaml', '/mnt/nodeB/cfg.yaml'])
   svc.start()
   ```
2. **Edit `config.yaml`** as usual. Updates propagate automatically to each path
   listed when the file changes.
3. **Stop the service** with `svc.stop()` once synchronisation is no longer required.


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
   layer = MarbleAutogradLayer(marble.brain, accumulation_steps=4)
   # "accumulation_steps" accumulates gradients over multiple backward passes
   # before applying them, enabling larger effective batch sizes.
   out = layer(torch.tensor(1.0, requires_grad=True))
   ```
5. **Inject MARBLE into any PyTorch model** with `attach_marble_layer` to run
   MARBLE side by side without altering the model output.

**Complete Example**
```python
# project4_autograd_challenge.py
from sklearn.datasets import load_digits
from config_loader import load_config
from marble_main import MARBLE
from marble_autograd import MarbleAutogradLayer
from marble import DataLoader
import pytorch_challenge

digits = load_digits(return_X_y=True)
cfg = load_config()
dataloader = DataLoader()  # digits are numeric, tokenizer not required
train_pairs = [
    (dataloader.encode(x), dataloader.encode(t))
    for x, t in zip(*digits)
]
marble = MARBLE(cfg['core'])
pretrained = pytorch_challenge.load_pretrained_model()
pytorch_challenge.run_challenge(train_pairs, pretrained_model=pretrained, cfg=cfg)

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
3. **Tokenize and train** using the helper functions. The dataset is first
   processed with the dataloader so tokens are encoded consistently:
   ```python
   dataset, vocab = advanced_gpt.load_text_dataset(cfg['gpt']['dataset_path'])
   token_seqs = [dataloader.encode_array(seq) for seq in dataset]
   advanced_gpt.train_advanced_gpt(
       [dataloader.decode_array(s) for s in token_seqs],
       vocab_size=len(vocab),
       block_size=cfg["gpt"]["block_size"],
       epochs=5,
   )
   ```
4. **Generate text** once training completes by calling `advanced_gpt.generate_text(marble.brain, 'Once upon a time')`.
5. **Optionally distill** the knowledge to a smaller network with `DistillationTrainer` by loading a saved model and training a student brain against it.

**Complete Example**
```python
# project5_gpt_training.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from tokenizer_utils import built_in_tokenizer
import advanced_gpt
import os, urllib.request

cfg = load_config()
tokenizer = built_in_tokenizer("byte_level_bpe")  # suitable for plain text
dataloader = DataLoader(tokenizer=tokenizer)
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

## Project 5b – RNN Sequence Modeling (Intermediate)

**Goal:** Train a recurrent neural network on a short text corpus.**

1. **Download the dataset** from Karpathy's char-rnn repository:
   ```python
   import requests
   url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
   text = requests.get(url, timeout=10).text[:1000]
   ```
2. **Prepare sequences** by mapping characters to integer IDs and slicing windows of ten characters each.
3. **Train** the network with the new `rnn` neuron type using the provided script.

**Complete Example**
```python
# project05b_rnn_sequence_modeling.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from tokenizer_utils import built_in_tokenizer
import requests
import numpy as np

cfg = load_config()
tokenizer = built_in_tokenizer("char_bpe")  # character-level BPE for tiny corpus
dataloader = DataLoader(tokenizer=tokenizer)
marble = MARBLE(cfg['core'])
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url, timeout=10).text[:1000]
ids = dataloader.tokenizer.encode(text).ids
examples = [
    (
        dataloader.encode_array(np.array(ids[i : i + 10], dtype=np.int32)),
        dataloader.encode(ids[i + 10]),
    )
    for i in range(len(ids) - 10)
]
marble.brain.train(examples[:500], epochs=1)
```
Run `python project05b_rnn_sequence_modeling.py` to train the simple RNN example.
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

To experiment with a differentiable approach set ``reinforcement_learning.algorithm``
to ``"policy_gradient"`` and run ``project06b_policy_gradient.py``. This file
trains ``MarblePolicyGradientAgent`` using a policy network wrapped with
``MarbleAutogradLayer``. Rewards should steadily increase across episodes.

**Complete Example**
```python
# project6_reinforcement_learning.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from reinforcement_learning import train_gridworld

cfg = load_config()
dataloader = DataLoader()  # reinforcement learning uses numeric states
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
from marble import DataLoader
from marble_utils import augment_image

cfg = load_config()
dataloader = DataLoader()  # image dataset, tokenizer unnecessary
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
   inputs = [dataloader.encode(x['image'].reshape(-1).numpy() / 255.0) for x in ds]
   ```
4. **Train** on these vectors with `learner.train(inputs)` and review `learner.history` to see how correlations strengthen or weaken connections.

**Complete Example**
```python
# project8_hebbian_learning.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from hebbian_learning import HebbianLearner

cfg = load_config()
dataloader = DataLoader()  # images require no tokenizer
marble = MARBLE(cfg['core'])
learner = HebbianLearner(marble.core, marble.neuronenblitz)
from datasets import load_dataset
ds = load_dataset('fashion_mnist', split='train')
inputs = [dataloader.encode(x['image'].reshape(-1).numpy() / 255.0) for x in ds]
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
   ```
4. **Wrap the dataset** with `FGSMDataset` to generate adversarial variants on the fly:
   ```python
   from adversarial_dataset import FGSMDataset
   model = ToyModel()
   adv_ds = FGSMDataset(ds, model, epsilon=0.05)
   real_values = [dataloader.encode(x[0]) for x in adv_ds]
   ```
5. **Construct an `AdversarialLearner`** and call `learner.train(real_values)` to alternate generator and discriminator updates.
6. **Optionally perform adversarial fine-tuning** of a standard PyTorch model using
   `train_with_adversarial_examples`:
   ```python
   from adversarial_learning import train_with_adversarial_examples, ToyModel
   model = ToyModel()
   train_with_adversarial_examples(model, adv_ds, epsilon=0.05, epochs=3)
   ```
7. **Sample new data** after training by passing random noise vectors to the generator's `dynamic_wander` method.

**Complete Example**
```python
# project9_adversarial_learning.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from adversarial_learning import AdversarialLearner

cfg = load_config()
dataloader = DataLoader()  # MNIST images do not need tokenization
generator = MARBLE(cfg['core']).neuronenblitz
discriminator = MARBLE(cfg['core']).neuronenblitz
learner = AdversarialLearner(generator, discriminator)
from datasets import load_dataset
ds = load_dataset('mnist', split='train')
real_values = [dataloader.encode(x['image'].reshape(-1).numpy() / 255.0) for x in ds]
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
   values = [dataloader.encode(img / 16.0) for img in digits.data]
   ```
4. **Train** using `auto.train(values)` and inspect `auto.history` to see reconstruction losses decreasing over epochs.

**Complete Example**
```python
# project10_autoencoder.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from autoencoder_learning import AutoencoderLearner

cfg = load_config()
dataloader = DataLoader()  # autoencoder uses numeric image data
marble = MARBLE(cfg['core'])
auto = AutoencoderLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
values = [dataloader.encode(img / 16.0) for img in digits.data]
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
   dataloader = DataLoader()
   labeled = [
       (dataloader.encode(x), dataloader.encode(t))
       for x, t in zip(X_train, y_train)
   ]
   unlabeled = [dataloader.encode(x) for x in X_unlabeled]
   ```
4. **Train** using these lists with `learner.train(labeled, unlabeled)` and inspect `learner.history` to gauge progress.

**Complete Example**
```python
# project11_semi_supervised.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from semi_supervised_learning import SemiSupervisedLearner

cfg = load_config()
dataloader = DataLoader()  # digits are numeric
marble = MARBLE(cfg['core'])
learner = SemiSupervisedLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X_train, X_unlabeled, y_train, _ = train_test_split(
    digits.data, digits.target, test_size=0.8, random_state=42)
labeled = [
    (dataloader.encode(x), dataloader.encode(t))
    for x, t in zip(X_train, y_train)
]
unlabeled = [dataloader.encode(x) for x in X_unlabeled]
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
from marble import DataLoader

cfg = load_config()
dataloader = DataLoader()  # federated learning on images
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
   sorted_samples = [
       (dataloader.encode(digits.data[i]), dataloader.encode(digits.target[i]))
       for i in sorted_idx
   ]
   ```
4. **Call** `learner.train(sorted_samples)` to progressively feed in harder examples and track progress via `learner.history`.

**Complete Example**
```python
# project13_curriculum.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from curriculum_learning import CurriculumLearner

cfg = load_config()
dataloader = DataLoader()  # numeric digits again
marble = MARBLE(cfg['core'])
learner = CurriculumLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
difficulty = digits.data.sum(axis=1)
sorted_idx = difficulty.argsort()
sorted_samples = [
    (dataloader.encode(digits.data[i]), dataloader.encode(digits.target[i]))
    for i in sorted_idx
]
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
   dataloader = DataLoader()
   tasks = []
   for digit in range(10):
       mask = digits.target == digit
       pairs = [
           (dataloader.encode(x), dataloader.encode(int(digit)))
           for x in digits.data[mask]
       ]
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
from marble import DataLoader
from meta_learning import MetaLearner

cfg = load_config()
dataloader = DataLoader()  # digits are not textual
marble = MARBLE(cfg['core'])
learner = MetaLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
tasks = []
for digit in range(10):
    mask = digits.target == digit
    pairs = [
        (dataloader.encode(x), dataloader.encode(int(digit)))
        for x in digits.data[mask]
    ]
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
   dl = DataLoader()
   new_pairs = [
       (dl.encode(x['image'].reshape(-1).numpy() / 255.0), dl.encode(x['label']))
       for x in ds
   ]
   ```
4. **Fine-tune** on these pairs by calling `learner.train(new_pairs)`.
5. **Tune `freeze_fraction`** to control how many synapses stay fixed during fine‑tuning and monitor `learner.history` to check performance on the new task.

**Complete Example**
```python
# project15_transfer.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from transfer_learning import TransferLearner

cfg = load_config()
dataloader = DataLoader()  # working with image tensors
base = MARBLE(cfg['core'])
learner = TransferLearner(base.core, base.neuronenblitz)
from datasets import load_dataset
ds = load_dataset('fashion_mnist', split='train')
new_pairs = [
    (dataloader.encode(x['image'].reshape(-1).numpy() / 255.0),
     dataloader.encode(x['label']))
    for x in ds
]
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
   dl = DataLoader()
   datasets = [
       [
           (dl.encode(x), dl.encode(t))
           for x, t in zip(load_digits().data, load_digits().target)
       ],
       [
           (dl.encode(x), dl.encode(t))
           for x, t in zip(load_iris().data, load_iris().target)
       ],
       [
           (dl.encode(x), dl.encode(t))
           for x, t in zip(load_wine().data, load_wine().target)
       ],
   ]
   ```
4. **Train sequentially** by calling `learner.train(data)` for each dataset and monitor reconstruction loss via `learner.history` to see how well the model retains previous knowledge.

**Complete Example**
```python
# project16_continual.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from continual_learning import ReplayContinualLearner

cfg = load_config()
dataloader = DataLoader()  # pure numeric features
marble = MARBLE(cfg['core'])
learner = ReplayContinualLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits, load_iris, load_wine
datasets = [
    [
        (dataloader.encode(x), dataloader.encode(t))
        for x, t in zip(load_digits().data, load_digits().target)
    ],
    [
        (dataloader.encode(x), dataloader.encode(t))
        for x, t in zip(load_iris().data, load_iris().target)
    ],
    [
        (dataloader.encode(x), dataloader.encode(t))
        for x, t in zip(load_wine().data, load_wine().target)
    ],
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
from marble import DataLoader
from imitation_learning import ImitationLearner

cfg = load_config()
dataloader = DataLoader()  # RL observations are numeric
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
   dl = DataLoader()
   values = [
       (dl.encode(t), dl.encode(p))
       for t, p in zip(data['T (degC)'].values, data['p (mbar)'].values)
   ]
   ```
4. **Train** by repeatedly calling `learner.train_step(value, target)` for the specified number of epochs and observe frequency error in `learner.history` to understand how phase alignment evolves.

**Complete Example**
```python
# project18_harmonic.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from harmonic_resonance_learning import HarmonicResonanceLearner
import urllib.request, zipfile, io, pandas as pd

cfg = load_config()
dataloader = DataLoader()  # time-series numeric data
marble = MARBLE(cfg['core'])
learner = HarmonicResonanceLearner(marble.core, marble.neuronenblitz)
url = "https://github.com/philipperemy/keras-tutorials/raw/master/resources/jena_climate_2009_2016.csv.zip"
with urllib.request.urlopen(url) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        zf.extractall()
data = pd.read_csv('jena_climate_2009_2016.csv')
values = [
    (dataloader.encode(t), dataloader.encode(p))
    for t, p in zip(data['T (degC)'].values, data['p (mbar)'].values)
]
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
   dl = DataLoader()
   values = [
       (dl.encode(x), dl.encode(t))
       for x, t in zip(digits.data, digits.target)
   ]
   ```
4. **Train** by iterating over `values` and calling `learner.train_step(v, t)` while monitoring `learner.history` and the synapse echo buffers to see how past activations influence current learning.

**Complete Example**
```python
# project19_synaptic_echo.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from synaptic_echo_learning import SynapticEchoLearner

cfg = load_config()
dataloader = DataLoader()  # digits dataset
marble = MARBLE(cfg['core'])
learner = SynapticEchoLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
values = [
    (dataloader.encode(x), dataloader.encode(t))
    for x, t in zip(digits.data, digits.target)
]
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
   dl = DataLoader()
   pairs = [
       (dl.encode(x), dl.encode(t))
       for x, t in zip(digits.data, digits.target)
   ]
   ```
4. **Train** with `learner.train(pairs)` and watch representation size via `core.rep_size` or `learner.history` to see when new dimensions are added.

**Complete Example**
```python
# project20_fractal.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from fractal_dimension_learning import FractalDimensionLearner

cfg = load_config()
dataloader = DataLoader()  # digits dataset
marble = MARBLE(cfg['core'])
learner = FractalDimensionLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
pairs = [
    (dataloader.encode(x), dataloader.encode(t))
    for x, t in zip(digits.data, digits.target)
]
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
   dl = DataLoader()
   examples = [
       (dl.encode(x), dl.encode(t))
       for x, t in zip(digits.data, digits.target)
   ]
   ```
4. **Train** by repeatedly calling `learner.train_step(inp, tgt)` for each pair and track phases in `learner.phases` to understand how the system evolves over time.

**Complete Example**
```python
# project21_quantum_flux.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from quantum_flux_learning import QuantumFluxLearner

cfg = load_config()
dataloader = DataLoader()  # digits dataset
marble = MARBLE(cfg['core'])
learner = QuantumFluxLearner(marble.core, marble.neuronenblitz)
from sklearn.datasets import load_digits
digits = load_digits()
examples = [
    (dataloader.encode(x), dataloader.encode(t))
    for x, t in zip(digits.data, digits.target)
]
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
from marble import DataLoader
from dream_reinforcement_learning import DreamReinforcementLearner

cfg = load_config()
dataloader = DataLoader()  # numeric RL data
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
   dl = DataLoader()
   examples = [
       (dl.encode(x['image'].reshape(-1).numpy() / 255.0), dl.encode(x['label']))
       for x in ds
   ]
   ```
5. **Train** by providing these examples to `learner.train(examples, epochs=5)`. All paradigms run sequentially, leveraging interconnection synapses so multiple cores behave as one integrated system.

**Complete Example**
```python
# project23_omni.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from core_interconnect import interconnect_cores
from omni_learning import OmniLearner

cfg = load_config()
dataloader = DataLoader()  # omni-learning uses image dataset
cores = [MARBLE(cfg['core']).core for _ in range(3)]
combined = interconnect_cores(cores)
neuronenblitz = MARBLE(cfg['core']).neuronenblitz
learner = OmniLearner(combined, neuronenblitz)
from datasets import load_dataset
ds = load_dataset('mnist', split='train')
examples = [
    (dataloader.encode(x['image'].reshape(-1).numpy() / 255.0),
     dataloader.encode(x['label']))
    for x in ds
]
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
4. **Inspect decisions** with `learner.explain(index, with_gradients=True)` to
   see how each context feature influenced the chosen weights. The method
   returns the original context and weights along with gradient-based
   contributions for every learner.

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
   dl = DataLoader()
   samples = [
       (dl.encode(x), dl.encode(y))
       for x, y in zip(ds.data[:, 0], ds.target)
   ]
   ```
4. **Train** using `learner.train(samples, epochs=2)` and monitor `learner.history` for the squared error over time.

**Complete Example**
```python
# project24_cwfl.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from continuous_weight_field_learning import ContinuousWeightFieldLearner
from sklearn.datasets import load_diabetes

cfg = load_config()
dataloader = DataLoader()  # diabetes regression
marble = MARBLE(cfg['core'])
learner = ContinuousWeightFieldLearner(marble.core, marble.neuronenblitz)
ds = load_diabetes()
samples = [
    (dataloader.encode(x), dataloader.encode(y))
    for x, y in zip(ds.data[:, 0], ds.target)
]
learner.train(samples, epochs=2)
```
Run `python project24_cwfl.py` to see the field adapt across the dataset.

## Project 24b – Phase-Gated Neuronenblitz (Experimental)

**Goal:** Explore oscillatory modulation of synaptic updates.**

1. **Enable phase gating** by setting `neuronenblitz.phase_rate` and
   `neuronenblitz.phase_adaptation_rate` in `config.yaml`.
2. **Instantiate MARBLE** as usual and train on a small numeric dataset:
   ```python
   from config_loader import load_config
   from marble_main import MARBLE
   from marble import DataLoader

   cfg = load_config()
    dataloader = DataLoader()  # simple numeric demo
   marble = MARBLE(cfg['core'])
   examples = [
       (dataloader.encode(0.1), dataloader.encode(0.2)),
       (dataloader.encode(0.3), dataloader.encode(0.5)),
   ]
   marble.neuronenblitz.train(examples, epochs=3)
   ```
3. **Inspect phases** via `syn.phase` on any synapse to see how they
   gradually synchronise with the global oscillator.

Run `python project24b_phase_gated.py` to experiment with phase-based gating.

## Project 24c – Shortcut Synapse Formation (Experimental)

**Goal:** Automatically create direct connections along heavily traversed paths.**

1. **Set a threshold** by adjusting `neuronenblitz.shortcut_creation_threshold`
   in `config.yaml`. A value of ``0`` disables shortcut creation.
2. **Train or repeatedly call** `dynamic_wander` on the same inputs. Once the
   identical path has been taken enough times, MARBLE will insert a new synapse
   from the first to the last neuron of that path.
3. **Observe** the console for "Shortcut created" messages or inspect the core's
   synapse list to verify the new connection.

Run `python project24c_shortcuts.py` to watch shortcuts appear in a minimal
network.

## Project 24d – Chaotic Update Gating (Experimental)

**Goal:** Explore logistic-map modulation of weight updates.**

1. **Enable chaotic gating** by setting `neuronenblitz.chaotic_gating_enabled: true`
   in `config.yaml`. Adjust `chaotic_gating_param` and `chaotic_gate_init` to tune
   the behaviour.
2. **Train a small model** as in previous projects and observe
   `neuronenblitz.get_current_gate()` after each epoch to see how the gate evolves.

Run `python project24d_chaotic_gating.py` to test the effect on learning.

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
   dl = DataLoader()
   inputs = [dl.encode(x.reshape(-1).astype(float)) for x in digits.data]
   ```
4. **Train** using `learner.train(inputs, epochs=2)` and inspect
   `learner.schemas` to see the discovered patterns.

**Complete Example**
```python
# project25_nsi.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from neural_schema_induction import NeuralSchemaInductionLearner
from sklearn.datasets import load_digits

cfg = load_config()
dataloader = DataLoader()  # digit images flattened
marble = MARBLE(cfg['core'])
learner = NeuralSchemaInductionLearner(marble.core, marble.neuronenblitz)
digits = load_digits()
inputs = [dataloader.encode(x.reshape(-1).astype(float)) for x in digits.data]
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
   dl = DataLoader()
   samples = [dl.encode(v) for v in ds.data[:, 0]]
   ```
4. **Train** with `learner.train(samples, epochs=2)` and observe
   that `len(core.neurons)` grows as new concept nodes are inserted.

**Complete Example**
```python
# project26_cip.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from conceptual_integration import ConceptualIntegrationLearner
from sklearn.datasets import load_boston

cfg = load_config()
dataloader = DataLoader()  # numerical regression features
marble = MARBLE(cfg['core'])
learner = ConceptualIntegrationLearner(marble.core, marble.neuronenblitz,
                                       blend_probability=0.5,
                                       similarity_threshold=0.2)
ds = load_boston()
samples = [dataloader.encode(v) for v in ds.data[:, 0]]
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
   from marble import DataLoader

    dataloader = DataLoader()  # images as input
   train_pairs = load_hf_dataset(
       "mnist", "train[:100]", input_key="image", target_key="label",
       dataloader=dataloader,
   )
   ```
   If the dataset requires authentication place your Hugging Face token in
   `~/.cache/marble/hf_token`. The helper will log in automatically.
2. **Load a CSV dataset from a URL** using the `dataset_loader` utility:
   ```python
   from dataset_loader import load_dataset
   from marble import DataLoader

    dataloader = DataLoader()  # simple CSV data
   pairs = load_dataset(
       "https://example.com/data.csv",
       cache_dir="cached_datasets",
       dataloader=dataloader,
   )
   ```
3. **Load a zipped dataset**:
   ```python
   pairs = load_dataset("data.zip", dataloader=dataloader)
   ```
4. **Load a specific shard of a dataset** for distributed training:
   ```python
   pairs = load_dataset(
       "https://example.com/big.csv",
       num_shards=4,
       shard_index=1,
       dataloader=dataloader,
   )
   ```
   When defining parallel ``branches`` in a pipeline, ``BranchContainer`` sets
   ``num_shards`` to the number of branches and assigns each branch a unique
   ``shard_index`` automatically. This distributes large datasets across
   parallel pipelines with no extra configuration.
4. **Create and train a MARBLE system** using a pandas dataframe:
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
8. **Manage multiple instances** using the *Active Instance* selector. Create,
   duplicate or delete systems from the sidebar to compare configurations.
9. **Switch to Advanced mode** to access every function in
   ``marble_interface``. The playground displays each function's docstring and
   generates widgets for all parameters so you can call any operation directly.
10. **Search functions** using the filter boxes provided in Advanced mode to
    quickly locate operations by name.
11. **Build pipelines** on the *Pipeline* tab or with
   ``HighLevelPipeline``. Add steps from ``marble_interface`` directly as
   methods and access other modules using attribute notation like
   ``pipeline.plugin_system.load_plugins``. Press **Run Pipeline** to execute the
   sequence. The same pipelines can be saved as JSON and executed from the
   command line using ``python cli.py --pipeline mypipe.json`` or through the
   ``Pipeline``/``HighLevelPipeline`` classes in your own scripts.
   Nested modules are supported as well, so ``pipeline.marble_neuronenblitz.learning.enable_rl``
   appends a call to ``marble_neuronenblitz.learning.enable_rl``.  There is no
   hard limit on pipeline length, so you can chain together hundreds of steps if
   needed. Custom callables may be added as steps and any MARBLE instance returned
   (even inside tuples or dictionaries) becomes the active system for the
   following operations.
    - Set ``pipeline.async_enabled: true`` in ``config.yaml`` or pass
      ``async_enabled=True`` to ``HighLevelPipeline`` to overlap data loading and
      computation using ``asyncio``.
   - Provide ``pipeline.cache_dir`` to enable on-disk caching of step outputs.
12. **Edit pipelines interactively**. The Pipeline tab provides controls to add,
    reorder or remove steps with parameter widgets generated from function
    signatures. Save or load the pipeline as JSON and run it on CPU or GPU
    without leaving the browser.
13. **Dataset events surface as notifications**. Functions like
    ``dataset_loader.load_dataset`` publish ``dataset_load_start`` and
    ``dataset_load_end`` events so the GUI can display dataset progress alongside
    pipeline updates.
14. **Inspect pipeline steps** via the *Step Visualisation* expander. It shows
    each step's parameters and dataset summaries, streams live CPU/GPU memory
    usage while the pipeline runs and lets you download step details as JSON or
    CSV. These features work on both desktop and mobile layouts.
      The metrics dashboard tracks ``cache_hit`` and ``cache_miss`` so you can
      verify reuse. Clear the cache with ``HighLevelPipeline.clear_cache()`` when
      disk space runs low.
    - Assign each step a unique ``name`` and declare prerequisites with
      ``depends_on``. MARBLE resolves the ordering automatically using a
      topological sort. Cycles produce a ``ValueError`` describing the loop.

      ```python
      from pipeline import Pipeline

      pipe = Pipeline()
      pipe.add_step("load_data", module="marble_interface", name="load")
      pipe.add_step("train", module="marble_interface", name="train", depends_on=["load"])
      pipe.execute()
      ```

      Branching allows running alternative steps in parallel.  Create two
      branches and merge their results with ``add_branch``:

      ```python
      from pipeline import Pipeline

      pipe = Pipeline()
      pipe.add_branch(
          branches=[
              [{"func": "branch_a", "module": "my_steps"}],
              [{"func": "branch_b", "module": "my_steps"}],
          ],
          merge={"func": "merge_branches", "module": "my_steps"},
      )
      pipe.execute()
      ```

      Use branching when you want to compare preprocessing options or model
      architectures without running separate experiments.  Each branch receives
      its own execution context and the merge step combines outputs once every
      branch completes.

      When running on GPUs, branch execution can easily exhaust device memory.
      Pass ``max_gpu_concurrency`` to ``Pipeline.execute`` to bound the number of
      simultaneous GPU branches. Additional branches wait until prior ones
      finish, preventing out-of-memory errors while keeping CPU execution
      unrestricted.

      ```python
      pipe.execute(max_gpu_concurrency=1)
      ```

## Project: Macro Steps and Rollback

This project shows how to group several operations into a macro step and how to
roll back to a previous output when an experiment fails.

1. **Create a pipeline with a macro step.** The macro groups two steps that run
   sequentially on the available device (CPU or GPU).

   ```python
   from pipeline import Pipeline

   pipe = Pipeline()
   pipe.add_step("step_a", module="tests.dependency_steps", name="a")
   pipe.add_macro(
       "macro_bc",
       [
           {"func": "step_b", "module": "tests.dependency_steps", "name": "b"},
           {"func": "step_c", "module": "tests.dependency_steps", "name": "c"},
       ],
   )
   pipe.execute(cache_dir="cache")
   ```

2. **Simulate a failing experiment and roll back.** Modify a step so that it
   raises an error, then restore the previous output of step ``a`` and rerun the
   pipeline.

   ```python
   pipe.steps[1]["func"] = "failing_step"
   pipe.steps[1]["module"] = "tests.branching_steps"
   try:
       pipe.execute(cache_dir="cache")
   except RuntimeError:
       pipe.rollback("a", "cache")
       pipe.steps[1]["func"] = "step_b"
       pipe.steps[1]["module"] = "tests.dependency_steps"
   pipe.execute(cache_dir="cache")
   ```

The rollback removes any cached results after step ``a`` so the pipeline
re-executes from that point, preserving the previous computation.

      Troubleshooting:

      - ``Dependency cycle detected`` indicates a loop in the graph.
      - ``depends on unknown step`` means a dependency name was misspelled or
        the referenced step lacks a ``name``.
12. **View the core graph** on the *Visualization* tab. Press **Generate Graph**
   to see an interactive display of neurons and synapses.
13. **Inspect synaptic weights** on the *Weight Heatmap* tab. Set a maximum
   number of neurons and press **Generate Heatmap** to visualize connection
   strengths.
14. **Edit configuration** from the *Config Editor* tab. Specify any dot-
   separated parameter path and new value to update the YAML in place. Press
   **Reinitialize** to rebuild the system with the modified settings.
15. **Download or save** the current YAML configuration using the controls in
    the sidebar so you can preserve your settings between sessions.
16. **Consult the YAML manual** from the sidebar while adjusting parameters.
17. **Convert Hugging Face models** on the *Model Conversion* tab. Enter a
    model search query, preview the architecture and click **Convert to
    MARBLE** to initialize a system from the pretrained weights.
18. **Experiment with offloading** using the *Offloading* tab. Start a
    ``RemoteBrainServer`` or create a torrent client directly from the
    interface and attach it to the active MARBLE instance. Use the provided
    buttons to offload high‑attention lobes to the remote server or share them
    with peers via torrent.
19. **Monitor progress** on the *Metrics* tab. Loss, memory usage and other
    statistics are plotted live so you can observe training behaviour.
20. **Check system resources** on the *System Stats* tab. RAM and GPU usage are
    displayed so you can monitor consumption while experimenting.
21. **Tune neuromodulatory signals** on the *Neuromodulation* tab. Adjust the
    sliders for arousal, stress and reward or set an emotion string, then press
    **Update Signals** to modify MARBLE's internal context on the fly.
22. **Experiment with context history** by adjusting `context_history_size` and
    `context_embedding_decay` on the *Settings* tab. These options control how
    many neuromodulatory states influence wandering and how quickly older
    signals fade.
23. **Encourage emergent connections** by increasing
    `neuronenblitz.emergent_connection_prob` on the *Settings* tab. A value near
    `0.1` lets `dynamic_wander` occasionally create new random synapses, which
    can foster unexpected network behaviour.
24. **Detect wandering anomalies** by setting `wander_anomaly_threshold` under
    `neuronenblitz`. When path lengths deviate markedly from the running mean a
    warning is logged and the *Metrics* tab highlights the anomaly.
25. **Manage lobes** on the *Lobe Manager* tab. View attention scores for each
    lobe, create new lobes from selected neuron IDs, reorganize the current
    structure or apply self-attention updates with a single click.
26. **Read the documentation** on the *Documentation* tab. Every markdown file
    in the repository, including the architecture overview, configurable
    parameters list and machine learning handbook, can be opened here for quick
    reference.
25. **Browse source code** on the *Source Browser* tab. Select any module and
    click **Show Source** to view its implementation without leaving the
    playground.
26. **Run unit tests** on the *Tests* tab. Select one or more test files and
    click **Run Tests** to verify everything works as expected.
26. **Control asynchronous behaviour** on the *Async Training* tab. Start
    background training threads or enable auto-firing so MARBLE keeps learning
    while you explore other tabs. Use **Wait For Training** to block until the
    current session finishes or stop auto-firing at any time.
27. **Explore reinforcement learning** on the *RL Sandbox* tab. Set the grid
    size, number of episodes, step limit and optionally enable double
    Q-learning, then click **Run GridWorld** to train a
    `MarbleQLearningAgent`. The reward curve for each episode is displayed so
    you can observe learning progress.
28. **Tweak adaptive controllers** on the *Adaptive Control* tab. Inspect the
    meta-controller's loss history and adjust its parameters, review super
    evolution metrics and apply dimensional search or n-dimensional topology
    evaluations with a single click.
29. **Manage hybrid memory** on the *Hybrid Memory* tab. Initialize the vector
    and symbolic stores, add new key/value pairs, query for similar entries and
    prune older items without leaving the UI.

Uploaded datasets are previewed directly in the sidebar so you can verify their
contents before training. The currently active YAML configuration is also shown
in an expandable panel for quick reference. Use these previews to ensure your
data and settings are correct before experimenting.

The playground provides toggles for dreaming and autograd features so you can
experiment with MARBLE's advanced capabilities without writing code.

Additional experiments can enable **prioritized experience replay** by setting `neuronenblitz.use_experience_replay` to `true` in `config.yaml`. This stores recent training examples and replays them based on their errors, improving convergence on challenging datasets.

## Project 30 – Custom Plugin Modules

**Goal:** Extend MARBLE with additional neuron and synapse types.

1. Create a file `plugins/my_plugin.py` with a `register` function:
   ```python
   def register(register_neuron, register_synapse):
       register_neuron("my_neuron")
       register_synapse("my_synapse")
   ```
2. Edit `config.yaml` and add your plugin directory:
   ```yaml
   plugins:
     - ./plugins
   ```
3. Initialize MARBLE via `create_marble_from_config` and your new types will be
   available for use when modifying the core or Neuronenblitz.
4. After creating a `Neuronenblitz` instance you can activate plugins with
   `n_plugin.activate("my_plugin")`. The plugin's `activate(nb)` function
   receives the instance for unrestricted access.

The new *Global Workspace* plugin is loaded in the same way. Add
`global_workspace` settings to `config.yaml` and call
`n_plugin.activate("global_workspace")` to share broadcast messages between
plugins and components.

### Project 31 – Pipeline Step Plugin

**Goal:** Create a custom pipeline step using the plugin architecture.

1. **Write the plugin class** in ``plugins/double_step.py``:
   ```python
   from pipeline_plugins import PipelinePlugin, register_plugin
   import torch

   class DoubleStep(PipelinePlugin):
       def __init__(self, factor: float = 2.0):
           self.factor = factor

       def initialise(self, device, marble=None):
           self.device = device

       def execute(self, device, marble=None):
           return torch.ones(1, device=device) * self.factor

       def teardown(self):
           pass

   def register(reg):
       reg("double_step", DoubleStep)
   ```
2. **Load the plugin** before running a pipeline:
   ```python
   from pipeline_plugins import load_pipeline_plugins

   load_pipeline_plugins("plugins")
   ```
3. **Add a pipeline step** referencing the plugin:
   ```python
   from pipeline import Pipeline

   pipe = Pipeline([
       {"plugin": "double_step", "params": {"factor": 4}}
   ])
   print(pipe.execute()[0])  # tensor([4.])
   ```
4. **GPU support** is automatic.  If CUDA is available the ``device`` passed to
   ``initialise`` and ``execute`` will be ``"cuda"``; otherwise it defaults to
   ``"cpu"``.

### Using Attention Codelets

1. Define a codelet that returns an ``AttentionProposal``:
   ```python
   from attention_codelets import AttentionProposal, register_codelet

   def my_codelet():
       # score can be any float; higher values are more salient
       return AttentionProposal(score=1.0, content="Hello World")

   register_codelet(my_codelet)
   ```
2. Enable the plugin in ``config.yaml``:
   ```yaml
   attention_codelets:
     enabled: true
     coalition_size: 1
   ```
3. Activate both plugins before training:
   ```python
   import global_workspace
   import attention_codelets

   gw = global_workspace.activate(marble.brain, capacity=10)
  attention_codelets.activate(coalition_size=1)
  ```
4. Call ``attention_codelets.run_cycle()`` periodically to broadcast proposals
   through the workspace. Subscribers can listen via ``gw.subscribe``.

### Episodic Simulation for Planning

1. Ensure ``episodic_simulation`` is listed in ``plugins`` within ``config.yaml``
   and set ``episodic_sim_length`` to control how many past episodes are
   replayed.
2. Activate the plugin and run simulations before taking an action:
   ```python
   import episodic_memory
   import episodic_simulation

   mem = episodic_memory.EpisodicMemory()
  rewards = episodic_simulation.simulate(nb, mem, length=5)
  ```
  The list ``rewards`` contains predicted outcomes for the top episodes.

## Project 31 – Diffusion Core (Advanced)

**Goal:** Generate samples using MARBLE's dedicated diffusion engine.

1. **Enable diffusion settings** by adding the parameters ``diffusion_steps``,
   ``noise_start``, ``noise_end`` and ``noise_schedule`` under ``core`` in
   ``config.yaml``. The defaults perform ten linear denoising steps.
2. **Create the DiffusionCore** directly:
   ```python
   from config_loader import load_config
   from marble import DataLoader
   from diffusion_core import DiffusionCore

   cfg = load_config()
   dataloader = DataLoader()  # diffusion uses numeric inputs
   dcore = DiffusionCore(cfg["core"])
   output = dcore.diffuse(0.0)
   print("Final value", output)
   ```
3. **Store intermediate results** by enabling the ``hybrid_memory`` section in
   the configuration. Each diffusion step is embedded and saved so similar
   samples can be retrieved later.
4. **Offload** the core automatically when VRAM usage exceeds
   ``offload_threshold`` by passing a ``RemoteBrainClient`` to
   ``DiffusionCore``.
5. **Broadcast results** by setting ``workspace_broadcast: true`` under
   ``core``. Each call to ``diffuse`` then publishes the final sample through the
   Global Workspace so plugins can react.
6. **Enable advanced features** by adding ``activation_output_dir``, ``memory_system``,
   ``cwfl``, ``harmonic`` and ``fractal`` sections under ``core``. With these set
   the diffusion engine stores each step in hierarchical memory, trains the
   continuous weight field, adapts representations via harmonic resonance and
   fractal learning and finally writes an activation heatmap to the configured
   directory. The colour map used for this heatmap can be selected with
   ``activation_colormap``.

Running this project demonstrates the new ``DiffusionCore`` which integrates
Neuronenblitz wandering, hybrid memory and remote offloading to support
diffusion-based models in a fully data type agnostic way.


## Organising Multiple Experiments

You can define a list of experiment setups in `config.yaml` under the `experiments` section. Each entry overrides parameters for a single run. Invoke `create_marble_from_config` with a specific experiment name to load those settings.

## Using the Project Template

For a quick start you can copy the contents of the `project_template` directory.
It contains a ready-made `config.yaml` and `main.py` that follow the patterns
described in this tutorial. Place your dataset in the directory and run

```bash
python main.py
```

to train using the default settings.

## Tokenizing Text Input

MARBLE's ``DataLoader`` can transparently tokenize strings using the
``tokenizers`` library. Set ``tokenizer_type`` in the ``dataloader`` section of
``config.yaml`` (for example ``bert_wordpiece``) or provide a path via
``tokenizer_json`` to load a custom tokenizer. When enabled, all text data is
converted to token IDs before being fed into the network and decoded back after
inference. The ``tokenizer_vocab_size`` parameter controls the vocabulary size
when training a tokenizer from scratch using the YAML configuration.

To train a tokenizer yourself use ``tokenizer_utils.train_tokenizer`` and save
the resulting object to JSON:

```python
from tokenizer_utils import train_tokenizer, tokenizer_to_json
tok = train_tokenizer(["data/train.txt"], model="wordpiece", vocab_size=8000)
with open("my_tokenizer.json", "w") as f:
    f.write(tokenizer_to_json(tok))
```

Set ``tokenizer_json: my_tokenizer.json`` in ``config.yaml`` to use this
tokenizer for both training and inference.

### Enabling Round-Trip Integrity Checks

The ``dataloader`` section also exposes ``enable_round_trip_check`` and
``round_trip_penalty``. When enabled, each training example is encoded and then
decoded again. If the decoded value differs from the original, the specified
penalty is added to the training loss. This helps detect issues with
tokenization or custom serialization. ``track_metadata`` should remain ``true``
unless you have strict storage constraints because it ensures objects are
reconstructed with the correct Python type during decoding.

## Project 32 – BitTensorDataset Pipelines

**Goal:** Demonstrate training using the new `ContrastivePipeline` and `SemiSupervisedPairsPipeline` which rely on `BitTensorDataset` for arbitrary inputs.

1. **Create some example data**. For text inputs a simple file is enough:
   ```bash
   echo "hello" > data.txt
   echo "world" >> data.txt
   ```
2. **Train the contrastive pipeline** on this data:
   ```python
   from tokenizer_utils import built_in_tokenizer
   from marble_core import Core, DataLoader
   from contrastive_pipeline import ContrastivePipeline
   from tests.test_core_functions import minimal_params

   tok = built_in_tokenizer("bert_wordpiece", lowercase=True)
   tok.train(["data.txt"], vocab_size=20)
   core = Core(minimal_params())
   dl = DataLoader(tokenizer=tok)
   pipe = ContrastivePipeline(core, dataloader=dl, use_vocab=True)
   pipe.train(["hello", "world"], epochs=1)
   ```
3. **Use semi-supervised learning** with labeled and unlabeled pairs:
   ```python
   from semi_supervised_pairs_pipeline import SemiSupervisedPairsPipeline

   labeled = [("hello", "greeting"), ("world", "noun")]
   unlabeled = ["foo", "bar"]
   pipe = SemiSupervisedPairsPipeline(core, dataloader=dl, use_vocab=True)
   pipe.train(labeled, unlabeled, epochs=1)
   ```
   Both pipelines automatically convert objects to bit-level tensors so you can train on any Python data structure.

## Project 33 – Advanced BitTensor Pipelines

**Goal:** Explore imitation and fractal-dimension learning with arbitrary inputs.

1. **Train the imitation pipeline** on example demonstrations:
   ```python
   from imitation_pairs_pipeline import ImitationPairsPipeline

   core = Core(minimal_params())
   pipe = ImitationPairsPipeline(core, use_vocab=True)
   demos = [("left", "move"), ("right", "move")]
   pipe.train(demos, epochs=1)
   ```
2. **Model fractal dimension adjustments** using another dataset:
   ```python
   from fractal_dimension_pairs_pipeline import FractalDimensionPairsPipeline

  core = Core(minimal_params())
  pipe = FractalDimensionPairsPipeline(core, use_vocab=True)
  pairs = [("conceptA", "conceptB"), ("conceptC", "conceptD")]
  pipe.train(pairs, epochs=1)
  ```

## Project 34 – Additional BitTensor Pipelines

**Goal:** Try the latest pipelines that operate on bit-level representations.**

1. **Hebbian learning on arbitrary objects:**
   ```python
   from hebbian_pipeline import HebbianPipeline

   core = Core(minimal_params())
   pipe = HebbianPipeline(core, use_vocab=True)
   data = ["alpha", "beta", "gamma"]
   pipe.train(data, epochs=1)
   ```
2. **Fine-tune with transfer learning:**
   ```python
   from transfer_pairs_pipeline import TransferPairsPipeline

   core = Core(minimal_params())
   pipe = TransferPairsPipeline(core, freeze_fraction=0.3, use_vocab=True)
   examples = [("old", "new"), ("stale", "fresh")]
   pipe.train(examples, epochs=1)
   ```
3. **Experiment with quantum flux learning:**
   ```python
   from quantum_flux_pairs_pipeline import QuantumFluxPairsPipeline

   core = Core(minimal_params())
   pipe = QuantumFluxPairsPipeline(core, phase_rate=0.2, use_vocab=True)
   examples = [("up", "down"), ("left", "right")]
   pipe.train(examples, epochs=1)
   ```

## Project 35 – Curriculum and Harmonic Pipelines

**Goal:** Explore additional pipelines that work on bit-level data.**

1. **Train a curriculum learning pipeline:**
   ```python
   from curriculum_pairs_pipeline import CurriculumPairsPipeline
   from marble_core import Core
   from tests.test_core_functions import minimal_params

   core = Core(minimal_params())
   pipe = CurriculumPairsPipeline(core, use_vocab=True)
   pairs = [("easy", "task"), ("hard", "task")]
   pipe.train(pairs, epochs=1)
   ```
2. **Use harmonic resonance learning on pairs:**
   ```python
   from harmonic_resonance_pairs_pipeline import HarmonicResonancePairsPipeline

   core = Core(minimal_params())
   pipe = HarmonicResonancePairsPipeline(core, use_vocab=True)
   pairs = [("tone", "A"), ("tone", "B")]
   pipe.train(pairs, epochs=1)
   ```
3. **Integrate concepts from arbitrary objects:**
   ```python
   from conceptual_integration_pipeline import ConceptualIntegrationPipeline

   core = Core(minimal_params())
   pipe = ConceptualIntegrationPipeline(core, use_vocab=True)
   data = ["alpha", "beta", "gamma"]
   pipe.train(data, epochs=1)
   ```

## Project 36 – Theory of Mind Predictions

**Goal:** Train the theory of mind plugin to model agents.

1. **Enable the plugin** by setting `theory_of_mind.enabled: true` in `config.yaml`.
2. **Train** on small interaction traces:
   ```python
   from theory_of_mind import activate
   from config_loader import load_config
   cfg = load_config()
   tom = activate(hidden_size=8, num_layers=1, prediction_horizon=1)
   tom.train([(0, 1), (1, 0)], epochs=5)
   ```

## Project 37 – Predictive Coding Integration

**Goal:** Use predictive coding inside DiffusionCore.

1. **Activate predictive coding** by editing `config.yaml` under `predictive_coding`.
2. **Add the module** when constructing DiffusionCore:
   ```python
   from diffusion_core import DiffusionCore
   from predictive_coding import activate as pc_activate
   core = DiffusionCore(rep_size=4, predictive_coding_params={"num_layers": 2})
   core.predictive_coding = pc_activate(num_layers=2, latent_dim=4, learning_rate=0.001)
   ```

## Project 38 – Dataset Versioning and Replication

**Goal:** Track dataset changes and distribute files to remote workers.**

1. **Load a dataset** using ``dataset_loader.load_dataset``:
   ```python
   from dataset_loader import load_dataset
   pairs = load_dataset(
       "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
       input_col="sepal_length",
       target_col="species",
       limit=32,
   )
   ```
2. **Create a version diff** so modifications can be reproduced later:
   ```python
   from dataset_versioning import create_version, apply_version
   vid = create_version([], pairs, "versions")
   restored = apply_version([], "versions", vid)
   ```
3. **Export and replicate the dataset** to additional machines:
   ```python
   from dataset_loader import export_dataset
   from dataset_replication import replicate_dataset
   export_dataset(restored, "iris_subset.csv")
   replicate_dataset("iris_subset.csv", ["http://worker1:8000", "http://worker2:8000"])
   ```
Run ``python project38_dataset_tools.py`` to reproduce and distribute the dataset.

## Project 39 – Resource Monitoring

**Goal:** Track CPU, RAM and GPU usage during training.

1. **Profile current usage**:
   ```python
   from system_metrics import profile_resource_usage
   print(profile_resource_usage())
   ```
2. **Record metrics over time**:
   ```python
   from usage_profiler import UsageProfiler
   profiler = UsageProfiler(interval=1.0)
   profiler.start()
   # run training
   profiler.stop()
   ```
3. **Inspect the CSV** generated by the profiler to locate bottlenecks or memory leaks.

Run ``python project39_monitor.py`` to experiment with these helpers.

## Project 40 – Multiprocessing Dataset Sharing

This project demonstrates executing a pipeline step on multiple processes while
sharing a dataset between them.

1. **Set the worker count** via environment variable (optional):
   ```bash
   export MARBLE_WORKERS=4
   ```
2. **Prepare a dataset and manager**:
   ```python
   import torch
   from process_manager import SharedDataset, ProcessManager

   data = [torch.arange(8, dtype=torch.float32) for _ in range(16)]
   shared = SharedDataset.from_tensors(data)
   manager = ProcessManager(shared)
   ```
3. **Define and run a step**:
   ```python
   def normalise(x: torch.Tensor) -> torch.Tensor:
       return x / x.sum()

   out = manager.run(normalise)
   ```
4. **Debugging tips:** Enable fault handlers with ``PYTHONFAULTHANDLER=1`` and
   ensure the start method is ``spawn`` (handled automatically). If a worker
   appears hung, check that all queues are consumed and that no tensor is left on
   a closed device.

Executing the snippet above on a GPU-equipped machine automatically transfers
the tensors to CUDA for zero-copy processing.

## Project 41 – Online Chat Training with Ollama

**Goal:** Continuously fine-tune a MARBLE core using recent chat exchanges served by Ollama.**

1. **Download a seed conversation dataset** to initialise history:
   ```python
   from datasets import load_dataset
   ds = load_dataset("OpenAssistant/oasst1", split="train[:5]")
   history = [
       {"role": m["role"], "content": m["content"]}
       for m in ds[0]["messages"]
   ]
   ```
2. **Instantiate MARBLE and connect to Ollama**:
   ```python
   from config_loader import load_config
   from marble_main import MARBLE
   from ollama_interop import chat_with_history

   cfg = load_config()
   marble = MARBLE(cfg["core"])
   response, history = chat_with_history(
       marble.core, "marble-chat", "Hello there!", history, history_limit=8
   )
   print(response["message"]["content"])
   ```
3. **Train on updated history** by converting the collected messages into
   token sequences and feeding them to your learner (see projects above for
   dataset preparation techniques).

Run `python project41_chat_training.py` to experiment with live conversation fine-tuning.


## Project 42 – Dream Replay Buffer Consolidation

**Goal:** Use the dream replay buffer to consolidate memories from a small dataset.**

1. **Download a dataset** such as MNIST and prepare numeric examples:
   ```python
   from datasets import load_dataset
   from marble import DataLoader

   ds = load_dataset("mnist", split="train[:100]")
   dl = DataLoader()
   examples = [
       (dl.encode(x["image"].reshape(-1) / 255.0), dl.encode(x["label"]))
       for x in ds
   ]
   ```
2. **Enable dream replay** in the configuration:
   ```yaml
   dream_enabled: true
   dream_replay_buffer_size: 50
   dream_replay_batch_size: 8
   ```
3. **Train and trigger dreaming** to observe consolidation:
   ```python
   from config_loader import load_config
   from marble_main import MARBLE

   cfg = load_config()
   marble = MARBLE(cfg["core"])
   brain = marble.brain
   brain.train(examples, epochs=1)
   brain.start_dreaming(num_cycles=3, interval=2)
   ```

**Complete Example**
```python
# project42_dream_replay.py
from config_loader import load_config
from marble_main import MARBLE
from marble import DataLoader
from datasets import load_dataset

cfg = load_config()
marble = MARBLE(cfg["core"])
dl = DataLoader()
ds = load_dataset("mnist", split="train[:100]")
examples = [
    (dl.encode(x["image"].reshape(-1) / 255.0), dl.encode(x["label"]))
    for x in ds
]
brain = marble.brain
brain.train(examples, epochs=1)
brain.start_dreaming(num_cycles=3, interval=2)
```

### Project: Interactive Ollama Training Pipeline

This project demonstrates how to convert an Ollama model into a MARBLE `Core`
using the new `OllamaInteractiveTrainingPlugin` and fine‑tune it with
conversation pairs.

1. **Prepare a small dataset** of prompts and desired replies.  The example
   below uses the Stanford Sentiment Treebank which can be downloaded via the
   `datasets` library:
   ```python
   from datasets import load_dataset

   raw = load_dataset("sst2", split="train[:10]")
   pairs = [
       (rec["sentence"], "positive" if rec["label"] else "negative")
       for rec in raw
   ]
   ```
2. **Create a pipeline with the plugin and execute it**.  The plugin pulls the
   model from Ollama, converts it and trains on the provided pairs:
   ```python
   from pipeline import Pipeline
   from ollama_pipeline import OllamaInteractiveTrainingPlugin

   pipe = Pipeline([
       {"plugin": "ollama_interactive_train", "params": {"model": "llama3", "interactions": pairs}}
   ])
   result = pipe.execute()[0]
   print(result["responses"][0]["message"]["content"])
   ```
3. **Inspect the conversation history** stored in `result["history"]` to review
   how the assistant responded to each prompt after training.

The plugin uses GPU acceleration when available and falls back to CPU
otherwise.

## Project: Estimating Memory and Profiling Execution

This project shows how to plan memory usage and record the exact order of
executed steps.

1. **Construct a tiny pipeline** that allocates a tensor and requests a
   run profile:

   ```python
   from pipeline import Pipeline
   from memory_manager import MemoryManager

   steps = [
       {"module": "tests.test_resource_estimation", "func": "make_tensor", "params": {"size": 8}, "name": "alloc"}
   ]
   pipe = Pipeline(steps)
   mgr = MemoryManager()
   pipe.execute(memory_manager=mgr, run_profile_path="profile.json")
   print("Reserved bytes:", mgr.total_reserved())
   ```

2. **Inspect the profile** saved to ``profile.json``:

   ```python
   import json, pprint
   with open("profile.json") as f:
       pprint.pprint(json.load(f))
   ```

The profile lists each step with start and end times and the CPU or GPU
device used during execution.

3. **Evaluate with deterministic k-fold cross-validation** using the classic Iris dataset:

   ```python
   from sklearn.datasets import load_iris
   import torch
   from cross_validation import cross_validate

   # download and prepare dataset
   data = load_iris(as_frame=True)
   X = torch.tensor(data.data.values, dtype=torch.float32)
   y = torch.tensor(data.target.values, dtype=torch.float32)
   dataset = list(zip(X, y))

   def train(ds, device):
       # compute mean of targets as a tiny model
       return torch.mean(torch.stack([t for _, t in ds])).to(device)

   def metric(model, ds, device):
       preds = torch.full((len(ds),), model.item(), device=device)
       targets = torch.stack([t for _, t in ds]).to(device)
       return float(torch.mean((preds - targets) ** 2))

   scores = cross_validate(train, metric, dataset, folds=5, seed=0)
   print(scores)
   ```

   The Iris dataset originates from the UCI repository: https://archive.ics.uci.edu/dataset/53/iris

4. **Serve a model through the pipeline API**:

   ```python
   from pipeline import Pipeline
   from marble_core import Core, DataLoader
   from marble_neuronenblitz import Neuronenblitz
   from marble_brain import Brain
   from tests.test_core_functions import minimal_params

   core = Core(minimal_params())
   nb = Neuronenblitz(core)
   brain = Brain(core, nb, DataLoader())
   pipe = Pipeline([{ 'plugin': 'serve_model', 'params': {'host': 'localhost', 'port': 5082}}])
   info = pipe.execute(type('M', (), {'get_brain': lambda self=brain: brain, 'brain': brain})())[0]
   # interact with the server then stop
   info['server'].stop()
   ```

## Project: Tracking the MARBLE Topology in Kùzu

This project demonstrates how to mirror MARBLE's evolving network structure
into a persistent Kùzu graph database during training.

1. **Enable topology tracking** by editing `config.yaml` and setting:

   ```yaml
   topology_graph:
     enabled: true
     db_path: "topology.kuzu"
   ```

2. **Create and train a small system** on the Iris dataset. The following code
   downloads the dataset and performs a single training iteration while the
   Kùzu database records the generated neurons and synapses:

   ```python
   from sklearn.datasets import load_iris
   from config_loader import create_marble_from_config, load_config
   from marble_graph_builder import add_neuron_group, add_fully_connected_layer

   data = load_iris(as_frame=True)
   X = data.data.values.astype("float32")
   y = data.target.values.astype("float32")

   cfg = load_config("config.yaml")
   cfg["topology_graph"]["enabled"] = True
   marble = create_marble_from_config(config=cfg)

   inputs = add_neuron_group(marble.core, X.shape[1])
   add_fully_connected_layer(marble.core, inputs, 3)
   # pretend to train by just accessing data
   _ = list(zip(X, y))
   ```

3. **Inspect the graph** using Kùzu tools. Every neuron becomes a `Neuron`
   node and each synapse a `SYNAPSE` relationship:

   ```python
   from kuzu_interface import KuzuGraphDatabase
   db = KuzuGraphDatabase("topology.kuzu")
   print(db.execute("MATCH (n:Neuron) RETURN count(n) AS neurons"))
   print(db.execute("MATCH ()-[r:SYNAPSE]->() RETURN count(r) AS synapses"))
   ```

The Kùzu database updates live whenever new neurons or synapses are created or
the representation size changes, enabling external graph analytics while the
model trains.

