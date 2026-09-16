from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        # Allow CuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True

    # Avoid hash randomization affecting some libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int, base_seed: Optional[int] = 42) -> None:
    import numpy as np
    import torch

    seed = (42 if base_seed is None else base_seed) + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
