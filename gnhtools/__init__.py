"""
A new implementation of influence functions for PyTorch. Key differences with torch_influence:

- More memory efficient. Operates directly with model.parameters() and does not require to vectorize the whole model.
- Stochastic LiSSA implementation using either Adam or SGD. Feasible with a small batch size.
- Influence vector hashing with random projections
"""

__version__ = "0.1.0"

__all__ = [
    "BaseTask",
    "LiSSA",
    "GNHSketch",
    "GNHStats",
    "PBRF",
    "calc_hvp",
    "iters"
]

from gnhtools.base_task import BaseTask
from gnhtools.lissa import LiSSA
from gnhtools.sketch import GNHSketch, GNHStats
from gnhtools.pbrf import PBRF
import gnhtools.iters as iters
