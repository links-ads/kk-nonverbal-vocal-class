from typing import Type, Dict
from torch import nn
from .wav2vec2 import Wav2Vec2EncoderLayer
from .wavlm import WavLMEncoderLayer
from .whisper import WhisperEncoderLayer
from .hubert import HubertEncoderLayer
from .unispeech import UniSpeechEncoderLayer
from non_verbal_voc_class.configs import ModelConfig

class EncoderFactory:
    """Factory class for creating encoder layers based on model type."""
    
    ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {
        'wav2vec2': Wav2Vec2EncoderLayer,
        'wavlm': WavLMEncoderLayer,
        'whisper': WhisperEncoderLayer,
        'hubert': HubertEncoderLayer,
        'unispeech': UniSpeechEncoderLayer,
    }
    
    @classmethod
    def create_encoder(cls, config: ModelConfig, **kwargs) -> nn.Module:
        """
        Create an encoder layer instance based on model type.
        
        Args:
            model_type: Type of encoder ('wav2vec2', 'wavlm', 'whisper', 'hubert', 'unispeech')
            config: Configuration object containing encoder parameters
            **kwargs: Additional keyword arguments for encoder initialization
            
        Returns:
            Encoder layer instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_type = config.model_type.lower()

        if model_type not in cls.ENCODER_REGISTRY:
            supported_types = list(cls.ENCODER_REGISTRY.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported_types}")
        
        encoder_class = cls.ENCODER_REGISTRY[model_type]

        return encoder_class(config)

    @classmethod
    def get_supported_encoders(cls) -> list:
        """Get list of supported encoder types."""
        return list(cls.ENCODER_REGISTRY.keys())
    
    @classmethod
    def register_encoder(cls, model_type: str, encoder_class: Type[nn.Module]) -> None:
        """
        Register a new encoder type.
        
        Args:
            model_type: Name of the encoder type
            encoder_class: Encoder class to register
        """
        cls.ENCODER_REGISTRY[model_type.lower()] = encoder_class
