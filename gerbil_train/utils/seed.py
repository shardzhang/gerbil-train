"""Random seed helpers."""

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int = 42) -> None:
    """set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
