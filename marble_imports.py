import os
import sys
import torch
import numpy as np
import json
import tarfile
import tempfile
import requests
import time
from pathlib import Path
from tqdm.notebook import tqdm
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import pickle
import zlib
import random
import math
import sympy as sp
import threading
from datetime import datetime
# Determine CUDA availability and set up array module
try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()
    cp = _cp
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    import numpy as _cp
    cp = _cp
    cp.asnumpy = lambda x: x
    CUDA_AVAILABLE = False

DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
import torch.nn as nn
