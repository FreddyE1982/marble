# HighLevelPipeline Guide

The `HighLevelPipeline` class provides a fluent Python interface for building and executing complex MARBLE workflows. It exposes every function in
`marble_interface` and any other repository module as a chainable method.
Dataset arguments are converted to `BitTensorDataset` automatically so each step
receives inputs in a consistent format. The pipeline keeps track of the active
`MARBLE` instance and passes it between steps whenever one is returned.

---

## Who Should Use HighLevelPipeline

* **Researchers** who want to prototype MARBLE features quickly without writing
  boilerplate code for data loading and system initialisation.
* **Practitioners** building repeatable training workflows that mix multiple
  modules or paradigms, such as reinforcement learning with curriculum
  scheduling and custom plugins.
* **Educators** demonstrating MARBLE capabilities in a classroom or tutorial
  setting. The fluent syntax keeps notebooks concise and focused on the high
  level concepts.

---

## How to Use HighLevelPipeline

1. **Create a pipeline instance**
   ```python
   from highlevel_pipeline import HighLevelPipeline
   hp = HighLevelPipeline()
   ```
2. **Chain operations** using attribute access. Functions from
   `marble_interface` are added directly while any repository module can be
   accessed via dot notation.
   ```python
   hp.new_marble_system(config_path="config.yaml")
   hp.plugin_system.load_plugins(dirs=["plugins"])
   hp.train_marble_system(train_examples=[(0, 1)], epochs=5)
   ```
3. **Execute the pipeline**. The active `MARBLE` instance is returned along with
   results from each step.
   ```python
   marble, results = hp.execute()
   ```
4. **Save or load** pipelines using JSON for reproducibility.
   ```python
   hp.save_json("workflow.json")
   # later
   with open("workflow.json", "r", encoding="utf-8") as f:
       restored = HighLevelPipeline.load_json(f)
   ```
5. **Custom callables** can be inserted when more control is required.
   ```python
   def print_summary(marble=None):
       print(marble.summary())
   hp.add_step(print_summary)
   ```

---

# HighLevelPipeline Tutorial

The following twenty projects walk through every major feature of `HighLevelPipeline`.
Each project includes exact commands to download datasets and the full pipeline code.
Datasets can be stored anywhere; adjust paths as needed.

## Project 1 – Basic Regression

1. **Download data**
   ```bash
   curl -L -o wine.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
   ```
2. **Build and run pipeline**
   ```python
   from highlevel_pipeline import HighLevelPipeline
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="wine.csv", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 2 – Text Classification with Evaluation

1. **Download dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("ag_news").save_to_disk("ag_news")
PY
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="ag_news/train", epochs=3)
   hp.evaluate_marble_system(validation_examples="ag_news/test")
   marble, results = hp.execute()
   ```

## Project 3 – Automatic Dataset Conversion

1. **Download CIFAR10**
   ```bash
   curl -L -o cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   tar -xzf cifar-10-python.tar.gz
   ```
2. **Use list of pairs**
   ```python
   import pickle
   with open("cifar-10-batches-py/data_batch_1", "rb") as f:
       batch = pickle.load(f, encoding="bytes")
   pairs = list(zip(batch[b"data"], batch[b"labels"]))

   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples=pairs, epochs=2)
   marble, _ = hp.execute()
   ```

## Project 4 – Custom BitTensorDataset Parameters

1. **Download IMDB**
   ```bash
   curl -L -o imdb.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
   tar -xzf imdb.tar.gz
   ```
2. **Configure conversion**
   ```python
   hp = HighLevelPipeline()
   hp.set_bit_dataset_params(min_word_length=2, min_occurrence=1)
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="aclImdb/train", epochs=4)
   marble, _ = hp.execute()
   ```

## Project 5 – Register Additional Dataset Arguments

1. **Download TREC dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("trec").save_to_disk("trec")
PY
   ```
2. **Register `validation_examples`**
   ```python
   hp = HighLevelPipeline()
   hp.register_data_args("validation_examples")
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="trec/train", epochs=3)
   hp.evaluate_marble_system(validation_examples="trec/test")
   marble, _ = hp.execute()
   ```

## Project 6 – Saving and Loading Pipelines

1. **Create and save**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="trec/train", epochs=1)
   hp.save_json("hp.json")
   ```
2. **Load and run**
   ```python
   with open("hp.json", "r", encoding="utf-8") as f:
       hp2 = HighLevelPipeline.load_json(f)
   marble, _ = hp2.execute()
   ```

## Project 7 – Adding Custom Callables

1. **Download synthetic data**
   ```bash
   python synthetic_dataset.py --output synth.pkl
   ```
2. **Add callable step**
   ```python
   def print_loss(marble=None, results=None):
       print("Loss", results[-1])

   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="synth.pkl", epochs=2)
   hp.add_step(print_loss)
   marble, results = hp.execute()
   ```

## Project 8 – Plugin System Integration

1. **Prepare plugin** (assumes custom plugin directory)
   ```bash
   mkdir -p plugins
   # place your plugin code in plugins/my_plugin.py
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.plugin_system.load_plugins(dirs=["plugins"])
   hp.train_marble_system(train_examples="synth.pkl", epochs=2)
   marble, _ = hp.execute()
   ```

## Project 9 – Reinforcement Learning

1. **Install gym**
   ```bash
   pip install gym
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.enable_marble_rl()
   hp.marble_select_action(state=0, n_actions=2)
   hp.marble_update_q(state=0, action=1, reward=1.0, next_state=1, done=False)
   marble, _ = hp.execute()
   ```

## Project 10 – Imitation Learning

1. **Generate expert data**
   ```bash
   python scripts/generate_expert.py --env CartPole-v1 --output expert.pkl
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="expert.pkl", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 11 – Curriculum Learning

1. **Download CIFAR100**
   ```bash
   curl -L -o cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
   tar -xzf cifar-100-python.tar.gz
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.curriculum_train_marble_system(train_examples="cifar-100-python", epochs=10)
   marble, _ = hp.execute()
   ```

## Project 12 – Distillation Training

1. **Download teacher weights**
   ```bash
   curl -L -o teacher.pkl https://example.com/teacher.pkl
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.distillation_train_marble_system(train_examples="cifar-10-batches-py", teacher="teacher.pkl", epochs=3)
   marble, _ = hp.execute()
   ```

## Project 13 – Semi-Supervised Learning

1. **Download SVHN**
   ```bash
   curl -L -o svhn.tar.gz http://ufldl.stanford.edu/housenumbers/train_32x32.mat
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.semi_supervised_pairs_pipeline.train(train_examples="train_32x32.mat", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 14 – Autoencoder Training

1. **Download MNIST**
   ```bash
   curl -L -o mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_autoencoder(values=[0.1, 0.2, 0.3], epochs=3)
   marble, _ = hp.execute()
   ```

## Project 15 – Diffusion Pairs Pipeline

1. **Download CelebA sample**
   ```bash
   curl -L -o celeba.zip https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
   unzip celeba.zip
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.diffusion_pairs_pipeline.DiffusionPairsPipeline(core=marble.core).train(pairs=[(0,0)], epochs=1)
   marble, _ = hp.execute()
   ```

## Project 16 – Transfer Learning

1. **Download pretrained weights**
   ```bash
   curl -L -o resnet.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="cifar-10-batches-py", pretrained="resnet.pth", epochs=2)
   marble, _ = hp.execute()
   ```

## Project 17 – Meta Learning

1. **Download Omniglot**
   ```bash
   curl -L -o omniglot.zip https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
   unzip omniglot.zip -d omniglot
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.meta_learning.meta_train(dataset="omniglot", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 18 – Remote Offloading

1. **Start server** (in another terminal)
   ```bash
   python remote_offload.py --server &
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.remote_offload.RemoteBrainClient(address="http://localhost:8080").offload(marble.core)
   marble, _ = hp.execute()
   ```

## Project 19 – Metrics and Evaluation

1. **Download Fashion-MNIST**
   ```bash
   curl -L -o fashion-mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="fashion-mnist.npz", epochs=2)
   hp.metrics_dashboard.compute_metrics(marble=marble)
   marble, _ = hp.execute()
   ```

## Project 20 – Full Feature Chain

1. **Prepare COCO captions**
   ```bash
   curl -L -o coco2017.tar.gz http://images.cocodataset.org/zips/val2017.zip
   tar -xzf coco2017.tar.gz
   ```
2. **Complex pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.set_bit_dataset_params(min_word_length=3)
   hp.register_data_args("validation_examples")
   hp.new_marble_system(config_path="config.yaml")
   hp.plugin_system.load_plugins(dirs=["plugins"])
   hp.train_marble_system(train_examples="val2017", epochs=2)
   hp.evaluate_marble_system(validation_examples="val2017")
   hp.save_json("full.json")
   marble, results = hp.execute()
   ```

These exercises show how each feature fits together. Combine them to build your own complex pipelines.
