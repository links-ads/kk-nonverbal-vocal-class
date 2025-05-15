from .audio_classifier_model import AudioModelForClassification
from .old_models import make_model, HParams
from transformers import PretrainedConfig, PreTrainedModel
from torch import Tensor

class ModelsFactory:
    @staticmethod
    def create_model(model_type: str, model_config: PretrainedConfig, label_weights: Tensor) -> PreTrainedModel:
        """
        Factory function to get the appropriate model based on the model type.
    
        Parameters:
        ----------
        model_type (str): The type of the model. Must be one of "audio" or "multimodal".
        model_config (PretrainedConfig): The configuration for the model.

        Returns:
        --------
        PreTrainedModel: An instance of the model based on the model type.

        Raises:
        -------
        ValueError: If the model_type is unknown.
        """
        if model_type == "audio":
            return AudioModelForClassification(model_config)
        elif model_type == "old_audio":
            hparams = HParams(
                LABEL_WEIGHTS=label_weights,
                **model_config.old_model_kwargs
            )
            return make_model(hparams)
        else:
            raise ValueError(f"Unknown model type: {model_type}")