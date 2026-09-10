from utils.seed import set_seed

set_seed(42)

import random
import numpy as np
import torch

print("Python:", random.random())
print("NumPy :", np.random.rand())
print("Torch :", torch.rand(1))

if torch.cuda.is_available():
    print("CUDA  :", torch.rand(1, device="cuda"))

set_seed(42)
a = torch.rand(5)

set_seed(42)
b = torch.rand(5)

print("Same:", torch.equal(a, b))