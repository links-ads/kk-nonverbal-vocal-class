from transformers.utils import ModelOutput
from typing import Protocol

class BaseAudioModel(Protocol):
    """
    Base class for audio models. All audio models should inherit from this class.

    Supported audio models:
    --------------------------
        - Wav2Vec2
        - WavLM
        - Whisper
        
    Methods:
    --------
        freeze_parameters() -> None:
            Freezes the parameters (weights) of the audio model.
        freeze_feature_encoder() -> None:
            Freezes the feature encoder of the audio model.
        forward(*args, **kwargs) -> ModelOutput:
            Defines the computation performed at every call.

    """
    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the model.
        In this way, the parameters (weights) will not be updated during training.
        """
        ...

    def freeze_feature_encoder(self) -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder.
        In this way, its parameters (weights) will not be updated during training.
        """
        ...

    def forward(self, *args, **kwargs) -> ModelOutput:
        """
        Forward pass of the model.
        """
        ...