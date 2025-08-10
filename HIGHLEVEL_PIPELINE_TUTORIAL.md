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
4. **Duplicate or inspect** pipelines as needed.
   ```python
   clone = hp.duplicate()
   print(hp.describe())
   ```
5. **Save or load** pipelines using JSON for reproducibility.
   ```python
   hp.save_json("workflow.json")
   # later
   with open("workflow.json", "r", encoding="utf-8") as f:
       restored = HighLevelPipeline.load_json(f)
   ```
6. **Execute specific steps** to debug a workflow.
   ```python
  marble, result = hp.run_step(0)
  marble, intermediate = hp.execute_until(1)
  marble, tail = hp.execute_from(1)
  for m, res in hp.execute_stream():
      print("step result", res)
  ```
7. **Modify steps** by replacing functions or updating parameters.
   ```python
   hp.replace_step(0, some_other_func)
   hp.update_step_params(1, epochs=5)
   ```
8. **Custom callables** can be inserted when more control is required.
   ```python
   def print_summary(marble=None):
       print(marble.summary())
   hp.add_step(print_summary)
   ```

---

# HighLevelPipeline Tutorial

The following projects demonstrate how to create diverse MARBLE workflows using
`HighLevelPipeline`. Each project builds on real datasets that can be downloaded
with the commands shown. Replace the dataset paths with your preferred storage
location.

## Project 1 – Wine Quality Regression

1. **Download data**
   ```bash
   curl -L -o wine.csv https://archive.ics.uci.edu/static/public/186/wine+quality.zip
   unzip wine.csv
   ```
2. **Create pipeline**
   ```python
   from highlevel_pipeline import HighLevelPipeline
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="winequality-red.csv", epochs=10)
   marble, _ = hp.execute()
   ```

## Project 2 – AG NEWS Text Classification

1. **Download dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("ag_news").save_to_disk("ag_news")
PY
   ```
2. **Build pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="ag_news/train", epochs=3)
   marble, _ = hp.execute()
   ```

## Project 3 – CIFAR10 Image Classification

1. **Download dataset**
   ```bash
   python scripts/download_cifar10.py --output data/cifar10
   ```
2. **Setup pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="data/cifar10/cifar-10-batches-py", epochs=10)
   marble, _ = hp.execute()
   ```

## Project 4 – IMDB Sentiment Analysis

1. **Download data**
   ```bash
   python scripts/download_imdb.py --output data/imdb
   ```
2. **Pipeline steps**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="data/imdb/train.csv", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 5 – Boston Housing Regression

1. **Download data**
   ```bash
   curl -L -o housing.csv https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
   ```
2. **Build pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="housing.csv", epochs=15)
   marble, _ = hp.execute()
   ```

## Project 6 – Yahoo Answers Topic Classification

1. **Download dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("yahoo_answers_topics").save_to_disk("yahoo")
PY
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="yahoo/train", epochs=4)
   marble, _ = hp.execute()
   ```

## Project 7 – WikiText Language Modeling

1. **Download dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("wikitext", "wikitext-2-raw-v1").save_to_disk("wikitext2")
PY
   ```
2. **Create pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="wikitext2/train", epochs=3)
   marble, _ = hp.execute()
   ```

## Project 8 – IWSLT Machine Translation

1. **Download dataset**
   ```bash
   pip install datasets
   python - <<'PY'
from datasets import load_dataset
load_dataset("iwslt2017", "iwslt2017-de-en").save_to_disk("iwslt_de_en")
PY
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="iwslt_de_en/train", epochs=6)
   marble, _ = hp.execute()
   ```

## Project 9 – CartPole Reinforcement Learning

1. **Install environment**
   ```bash
   pip install gym
   ```
2. **Build pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.marble_neuronenblitz.learning.enable_rl(nb=None)
   hp.train_marble_system(train_examples=["CartPole-v1"], epochs=20)
   marble, _ = hp.execute()
   ```

## Project 10 – MNIST Autoencoder

1. **Download dataset**
   ```bash
   curl -L -o mnist.zip https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="mnist.npz", epochs=10)
   marble, _ = hp.execute()
   ```

## Project 11 – CelebA Diffusion Model

1. **Download data**
   ```bash
   curl -L -o celeba.zip https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
   unzip celeba.zip
   ```
2. **Build pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="img_align_celeba", epochs=20)
   marble, _ = hp.execute()
   ```

## Project 12 – STL10 Contrastive Learning

1. **Download dataset**
   ```bash
   curl -L -o stl10.zip https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
   tar -xzf stl10.zip
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="stl10_binary", epochs=15)
   marble, _ = hp.execute()
   ```

## Project 13 – CIFAR100 Curriculum Learning

1. **Download dataset**
   ```bash
   curl -L -o cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
   tar -xzf cifar-100-python.tar.gz
   ```
2. **Create pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="cifar-100-python", epochs=12)
   marble, _ = hp.execute()
   ```

## Project 14 – Imitation Learning with Gym Expert Data

1. **Prepare data**
   ```bash
   pip install gym
   python scripts/generate_expert.py --env CartPole-v1 --output expert.pkl
   ```
2. **Pipeline steps**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="expert.pkl", epochs=10)
   marble, _ = hp.execute()
   ```

## Project 15 – Transfer Learning from ImageNet

1. **Download pretrained weights**
   ```bash
   curl -L -o resnet.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="cifar-10-batches-py", pretrained="resnet.pth", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 16 – Semi-Supervised Learning with SVHN

1. **Download data**
   ```bash
   curl -L -o svhn.tar.gz http://ufldl.stanford.edu/housenumbers/train_32x32.mat
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="train_32x32.mat", epochs=8)
   marble, _ = hp.execute()
   ```

## Project 17 – Federated Learning Fashion-MNIST

1. **Download dataset**
   ```bash
   curl -L -o fashion-mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
   ```
2. **Pipeline steps**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="fashion-mnist.npz", epochs=5)
   marble, _ = hp.execute()
   ```

## Project 18 – Adversarial Training on MNIST

1. **Download dataset**
   ```bash
   curl -L -o mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
   ```
2. **Build pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="mnist.npz", epochs=10, adversarial=True)
   marble, _ = hp.execute()
   ```

## Project 19 – Meta Learning with Omniglot

1. **Download dataset**
   ```bash
   curl -L -o omniglot.zip https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
   unzip omniglot.zip -d omniglot
   ```
2. **Pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="omniglot", epochs=20)
   marble, _ = hp.execute()
   ```

## Project 20 – COCO Multimodal Training

1. **Download captions**
   ```bash
   curl -L -o coco2017.tar.gz http://images.cocodataset.org/zips/val2017.zip
   tar -xzf coco2017.tar.gz
   ```
2. **Create pipeline**
   ```python
   hp = HighLevelPipeline()
   hp.new_marble_system(config_path="config.yaml")
   hp.train_marble_system(train_examples="val2017", epochs=5)
   marble, _ = hp.execute()
   ```

These examples provide starting points for building your own workflows with
`HighLevelPipeline`. Adjust epochs, learning rates and parameters as needed.

## Saving and Resuming Pipelines

Long running experiments can be checkpointed and later resumed without
rebuilding the entire workflow. The `highlevel_pipeline_cli.py` script
provides two commands:

```bash
# Execute pipeline JSON and create a checkpoint
python highlevel_pipeline_cli.py checkpoint pipeline.json pipeline.chk --config config.yaml --device cpu

# Resume from a saved checkpoint
python highlevel_pipeline_cli.py resume pipeline.chk --config config.yaml --device cpu
```

Checkpoints store the pipeline steps and the `dataset_version` metadata so
repeated runs continue with the exact same dataset revision.
