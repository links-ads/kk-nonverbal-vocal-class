import torch
from typing import Protocol
from typing import Dict, List, Any
from torch import Tensor

class BaseCollator(Protocol):
    """
    Base class for collators.

    Supported collators:
    --------------------------
        - AudioCollator
        - MultiModalCollator
        
    Methods:
    --------
    __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]
        Method to be overridden by all collators to return a dictionary of collated features.
    """
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        ...