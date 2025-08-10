   python scripts/download_cifar10.py --output data/cifar10
   hp.train_marble_system(train_examples="data/cifar10/cifar-10-batches-py", epochs=10)
   python scripts/download_imdb.py --output data/imdb
   hp.train_marble_system(train_examples="data/imdb/train.csv", epochs=5)
1. **Download the dataset programmatically** using the helper script:
   ```bash
   python scripts/download_cifar10.py --output data/cifar10
   The script retrieves the CIFAR‑10 archive and extracts the training and test
   splits under `data/cifar10`.
import pickle
# Ensure the CIFAR-10 dataset is available using:
#   python scripts/download_cifar10.py --output data/cifar10
