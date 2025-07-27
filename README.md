# MARBLE

**Mandelbrot Adaptive Reasoning Brain-Like Engine**

The MARBLE system is a modular neural architecture that begins with a Mandelbrot-inspired seed and adapts through neuromodulatory feedback and structural plasticity. This repository contains the source code for the system along with utilities and configuration files.
It also includes a new unsupervised Hebbian learning module that ties together message passing in the Core with Neuronenblitz path exploration.
An autoencoder learning paradigm further reconstructs noisy inputs through Neuronenblitz wander paths integrated with the Core.
A semi-supervised paradigm leverages both labeled and unlabeled data by applying consistency regularization directly through Neuronenblitz.
A meta-learning module implements a Reptile-style algorithm allowing
rapid adaptation to new tasks by blending weights from short inner
training loops back into the main network.
Transfer learning is supported through a new learner that freezes a
fraction of synapses while fine-tuning on a different dataset.
Continual learning is enabled via a replay-based learner that revisits
previous examples to prevent catastrophic forgetting between tasks.
An ``OmniLearner`` paradigm seamlessly unifies all available learners so
that multiple approaches can train the same model in concert. Building on
this idea, the new ``UnifiedLearner`` introduces a gating network that
dynamically selects or blends paradigms for every training step based on
contextual cues. This meta-controller coordinates learning across all
modules while logging its decisions for later inspection.
The logged data can be analysed using ``UnifiedLearner.explain`` which
optionally returns gradient-based attributions showing how each context
feature influenced the gating weights.
Continuous Weight Field Learning introduces a variational method where each
input has its own smoothly varying weight vector generated on the fly.
Neural Schema Induction grows new neurons representing frequently repeated
reasoning patterns so the network can recall entire inference chains as single
concepts.
Conceptual Integration goes a step further by blending the representations of
two dissimilar neurons into a new "concept" neuron, allowing MARBLE to invent
abstract ideas not present in the training data.
The Hybrid Memory Architecture augments MARBLE with a vector store and symbolic
database so long conversations can be recalled accurately and hallucinations are
reduced.

MARBLE can train on datasets provided as lists of ``(input, target)`` pairs or using PyTorch-style ``Dataset``/``DataLoader`` objects. Each sample must expose an ``input`` and ``target`` field. After training and saving a model, ``Brain.infer`` generates outputs when given only an input value.
For quick experiments without external files you can generate synthetic regression pairs using ``synthetic_dataset.generate_sine_wave_dataset``.

## Command Line Interface

The ``cli.py`` script offers a convenient way to train and evaluate MARBLE directly
from the terminal.  It supports overriding key configuration parameters such as
learning rate scheduler and early stopping without editing ``config.yaml``.

Example usage:

```bash
python cli.py --config config.yaml --train data.csv --epochs 5 \
    --lr-scheduler cosine --scheduler-steps 10 --save marble.pkl
```

See ``python cli.py --help`` for the full list of options.

Any Python object can serve as an ``input`` or ``target`` because the built-in
``DataLoader`` serializes data through ``DataCompressor``. This makes it
possible to train on multimodal pairs such as text-to-image, image-to-text or
even audio and arbitrary byte blobs without additional conversion steps.

### Command Line Usage

Common tasks can be performed via ``cli.py``. Train a model, evaluate it and
export the core JSON with:

```bash
python cli.py --config path/to/cfg.yaml --train data.csv --epochs 10 \
    --export-core trained_core.json
```

Use ``--save`` to persist the entire MARBLE object with pickle and
``--export-core`` to write just the core for later loading via
``import_core_from_json``.

### Remote Inference API

MARBLE can be exposed through a lightweight HTTP API. Launch the server with:

```python
from marble_core import Core, DataLoader
from marble_neuronenblitz import Neuronenblitz
from marble_brain import Brain
from web_api import InferenceServer
from tests.test_core_functions import minimal_params

core = Core(minimal_params())
nb = Neuronenblitz(core)
brain = Brain(core, nb, DataLoader())
server = InferenceServer(brain)
server.start()
```

Query the API using ``curl``:

```bash
curl -X POST http://localhost:5000/infer -H 'Content-Type: application/json' \
     -d '{"input": 0.5}'
```

Call ``server.stop()`` to shut it down.

### Playground

An interactive Streamlit playground allows quick experimentation with all of
MARBLE's capabilities. Launch it from the repository root with:

```bash
streamlit run streamlit_playground.py
```

Upload CSV, JSON or ZIP datasets containing any mix of numbers, text, images or
audio. Provide a YAML configuration by path, file upload or inline text before
initializing the system. The inference panel accepts the same modalities so you
can explore how different data types influence the system in real time. Models
may be saved and loaded from the sidebar, and you can export or import the core
JSON for experimentation. Advanced mode displays function docstrings and
generates widgets for each parameter so every capability of the
``marble_interface`` can be invoked without writing code. Modules from the
repository are also exposed and you can construct a **pipeline** of function
calls that execute sequentially. This makes it possible to combine training,
evaluation and utility operations into a single workflow directly from the UI.
The playground now also includes a **Model Conversion** tab for loading any
pretrained model from the Hugging Face Hub and converting it into a MARBLE
system with one click.
Pipelines can be imported or exported as JSON and a **Custom Code** tab lets you
run arbitrary Python snippets with the active MARBLE instance.
Pipeline steps may also be reordered or removed directly from the UI so complex
workflows can be iterated on quickly.
Multiple MARBLE systems can be created in one session. Use the *Active Instance*
selector in the sidebar to switch between them, duplicate a system for
comparison or delete instances you no longer need.
The advanced interface now features a **Config Editor** tab where any
parameter from the YAML configuration can be modified on the fly.  Changes are
applied immediately and you can re-initialise the system with the updated
configuration without leaving the browser.
The sidebar now previews uploaded datasets and shows the active configuration
YAML so you can verify exactly what will be used for training and inference.
You can also **search** the Hugging Face Hub directly from the sidebar. Enter a
query, press **Search Datasets** and select a result to populate the dataset
name field without leaving the playground.
You can now download the currently active configuration or save it to a custom
path directly from the sidebar. Advanced mode also features search boxes for
quickly locating functions by name within ``marble_interface`` or any module.
The Model Conversion tab now supports searching the Hub for pretrained models so
they can be converted with a single click.
An additional **Visualization** tab renders an interactive graph of the core so
you can inspect neuron connectivity in real time. The new **Weight Heatmap** tab
displays synaptic strengths as a heatmap for deeper analysis. The sidebar also contains a
collapsible YAML manual for quick reference while experimenting.
The playground now includes an **Offloading** tab. This lets you start a
``RemoteBrainServer`` or create a ``RemoteBrainClient`` directly from the UI and
attach it to the running system. You can also spin up a torrent client with its
own tracker to distribute lobes among peers. High‑attention regions of the brain
may then be offloaded to the remote server or shared via torrent with a single
button press.
A dedicated **Metrics** tab graphs loss, memory usage and other statistics in
real time inside the browser. A **System Stats** tab displays current CPU and
GPU memory usage. Another **Documentation** tab provides quick access to the
README, YAML manual and full tutorial without leaving the playground. A **Tests**
tab lets you run the repository's pytest suite directly from the UI so you can
validate changes after each experiment.
The new **Neuromodulation** tab exposes sliders for arousal, stress and reward
signals along with an emotion field so you can tweak the brain's internal state
interactively.
A **Lobe Manager** tab displays all lobes with their current attention scores and
lets you create, organize or run self‑attention directly from the interface.
A new **Async Training** tab lets you launch background training threads and
enable MARBLE's auto-firing mechanism so learning continues while you explore
other features.
An **Adaptive Control** tab exposes the MetaParameterController and
SuperEvolutionController so you can fine-tune plasticity behaviour, inspect
parameter changes and run dimensional search or n-dimensional topology updates.
The interface now includes a **Hybrid Memory** tab as well. This lets you
initialize a vector store and symbolic memory, store new values, query for the
closest matches and prune old entries directly from the playground.

## Possible MARBLE Backcronyms

Below is a list of ideas explored when naming the project:

- Mandelbrot Adaptive Reasoning Brain-Like Engine
- Multi-Agent Reinforcement-Based Learning Environment
- Modular Architecture for Rapid Brain-Like Experimentation
- Memory-Augmented Recursive Bayesian Learning Engine
- Metaheuristic Adaptive Reinforcement-Based Learning Ecosystem
- Multimodal Analytical Response Behavior Learning Entity
- Machine-Augmented Reflective Belief Learning Engine
- Morphological Adaptive Robotics Brain-Like Executor
- Multi-sensory Associative Response and Behavior Learning Engine
- Matrix-Accelerated Reasoning Bot with Learning Enhancements
For a high level description of the system components and data flow see [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md). The YAML configuration format is documented in detail in [yaml-manual.txt](yaml-manual.txt).

## Hyperparameter Tuning Best Practices
Tuning MARBLE effectively requires balancing learning stability with network plasticity. Some general guidelines:

1. Start with `random_seed` fixed to make experiments repeatable.
2. Use `learning_rate` values between `0.001` and `0.05` for most tasks.
3. Increase `neurogenesis_factor` gradually when validation loss stalls.
4. Monitor `plasticity_threshold` as it controls when new connections form.
5. Enable `lr_scheduler` with `scheduler_gamma` around `0.99` for long runs.

These heuristics work well across the provided examples but every dataset benefits from its own small grid search.
When tuning a new task consider the following workflow:

1. Start with the default configuration and run a short training job to verify that the loss decreases.
2. Perform a coarse grid search over `learning_rate` and `lr_scheduler` settings using the helper functions in `hyperparameter_search.py`.
3. Once a stable range is found, explore `representation_size` and `message_passing_alpha` which strongly influence capacity and convergence speed.
4. Monitor GPU and CPU usage using the metrics dashboard to ensure batch size and dimensionality fit your hardware budget.
5. Keep `gradient_clip_value` low (around `1.0`) when experimenting with very large learning rates or aggressive neurogenesis.

Documenting the parameters of each run with the new experiment tracker makes it easy to compare results later.

## Troubleshooting
If training diverges or produces NaNs:

1. Verify the dataset is correctly formatted and free of missing values.
2. Lower `learning_rate` and check that `gradient_clip_value` is set.
3. Ensure message passing dropout is not too high for small graphs.
4. Use the metrics dashboard to watch memory usage spikes which may indicate a bug.

For CUDA related errors confirm that your GPU drivers and PyTorch build match.

## Additional Resources

* **Interactive notebook** – A Jupyter notebook located under ``notebooks/``
  demonstrates dataset loading and model training step by step.
* **Template repository** – The ``project_template`` directory provides a
  minimal project skeleton including a configuration file and entry point.
* **Reusable neuron and synapse templates** – Found in ``templates/`` to help
  kick start custom component development.

## Code Style
This repository includes a pre-commit configuration using **black**, **isort**
and **ruff**. After installing the `pre-commit` package run:

```bash
pre-commit install
```

to automatically format and lint changes before each commit.

\nMARBLE can be extended via a simple plugin system. Specify directories in the `plugins` list of the configuration and each module's `register` function will be invoked to add custom neuron or synapse types.
Neuronenblitz exposes a runtime plugin API. After creating a `Neuronenblitz` instance you may activate modules via `n_plugin.activate("my_plugin")`. The plugin\x27s `activate(nb)` function receives the instance and can freely read or modify any attributes or methods.
