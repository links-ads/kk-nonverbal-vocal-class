from typing import Protocol
from datasets import Dataset

class BasePreprocessor(Protocol):
    """
    Base class for preprocessors. All preprocessors should inherit from this class.

    Methods:
    --------
        preprocess(*args, **kwargs) -> None:
            Preprocesses the data.
    """
    def preprocess(self, *args, **kwargs) -> Dataset:
        """
        Preprocesses the data.
        """
        ...