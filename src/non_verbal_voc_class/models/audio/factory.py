from .audio_models import (
    BaseAudioModel,
    Wav2Vec2ModelWrapper, 
    WavLMModelWrapper, 
    WhisperModelWrapper
)
from transformers import PretrainedConfig

class AudioModelFactory:
    @staticmethod
    def create_audio_model(model_type: str, model_config: PretrainedConfig) -> BaseAudioModel:
        """
        Factory function to get the appropriate audio model wrapper based on the model type.
    
        Parameters:
        ----------
        model_type (str): The type of the audio model. Must be one of 'wav2vec2', 'wavlm', or 'whisper'.
        model_config (PretrainedConfig): The configuration for the model. Must be one of the model configs available (Wav2VecConfig, WavLMConfig, WhisperConfig).

        Returns:
        --------
        BaseAudioModel: An instance of the audio model wrapper based on the model type.

        Raises:
        -------
        ValueError: If the model_type is unknown.
        """
        if model_type == 'wav2vec2':
            return Wav2Vec2ModelWrapper(model_config)
        elif model_type == 'wavlm':
            return WavLMModelWrapper(model_config)
        elif model_type == 'whisper':
            return WhisperModelWrapper(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")