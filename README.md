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

Any Python object can serve as an ``input`` or ``target`` because the built-in
``DataLoader`` serializes data through ``DataCompressor``. This makes it
possible to train on multimodal pairs such as text-to-image, image-to-text or
even audio and arbitrary byte blobs without additional conversion steps.

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
The advanced interface now features a **Config Editor** tab where any
parameter from the YAML configuration can be modified on the fly.  Changes are
applied immediately and you can re-initialise the system with the updated
configuration without leaving the browser.
The sidebar now previews uploaded datasets and shows the active configuration
YAML so you can verify exactly what will be used for training and inference.
You can also **search** the Hugging Face Hub directly from the sidebar. Enter a
query, press **Search Datasets** and select a result to populate the dataset
name field without leaving the playground.
The Model Conversion tab now supports searching the Hub for pretrained models so
they can be converted with a single click.
An additional **Visualization** tab renders an interactive graph of the core so
you can inspect neuron connectivity in real time. The sidebar also contains a
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
A new **Async Training** tab lets you launch background training threads and
enable MARBLE's auto-firing mechanism so learning continues while you explore
other features.

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
