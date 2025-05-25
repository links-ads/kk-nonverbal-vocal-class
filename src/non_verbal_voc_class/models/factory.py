from torch import Tensor
from ..configs import ModelConfig
from .base_classifier import BaseClassifier
from .wav2vec2_classifier import Wav2Vec2Classifier
from .wavlm_classifier import WavLMClassifier
from .whisper_classifier import WhisperClassifier
from .hubert_classifier import HubertClassifier

class ModelFactory:
    @staticmethod
    def create_model(
        config: ModelConfig,
    ) -> BaseClassifier:
        
        model_type = config.model_type.lower()
        if model_type == 'wav2vec2':
            return Wav2Vec2Classifier(config)
        elif model_type == 'wavlm':
            return WavLMClassifier(config)
        elif model_type == 'whisper':
            return WhisperClassifier(config)
        elif model_type == 'hubert':
            return HubertClassifier(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")