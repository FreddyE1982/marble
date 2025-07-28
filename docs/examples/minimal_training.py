from marble_core import Core
from marble_neuronenblitz import Neuronenblitz
from dataset_loader import load_dataset

core = Core({"width": 4, "height": 4})
blitz = Neuronenblitz(core)

train_data = load_dataset("data/example.csv")
blitz.train(train_data, epochs=1)
