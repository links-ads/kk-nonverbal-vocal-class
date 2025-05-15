import torch
import torch.nn as nn

from .base_audio_model import BaseAudioModel
from transformers.utils import ModelOutput
from transformers import (
    WhisperModel,
    PreTrainedModel,
    WhisperConfig,
)

class WhisperModelWrapper(PreTrainedModel, BaseAudioModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.whisper = WhisperModel.from_pretrained(config._name_or_path)
        
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_parameters(self) -> None:
        """
        Calling this function will disable the gradient computation for all the parameters of the model.
        In this way, the parameters (weights) will not be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False

    def freeze_feature_encoder(self) -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder.
        In this way, its parameters (weights) will not be updated during training.
        """
        self.whisper.feature_extractor._freeze_parameters()

    def forward(self, input_features, attention_mask, labels=None) -> ModelOutput:
        """
        Forward pass of the model. 

        Parameters:
        ----------
            input_features (torch.Tensor): The input features of shape (B, N).
            attention_mask (torch.Tensor): The attention mask of shape (B, N).
            labels (torch.Tensor): The labels of shape (B, 1).

            Where:
                B: Batch size.
                N: Number of feature vectors.

        Returns:
        --------
            ModelOutput: The output of the model containing the hidden states, and the loss (for finetuning). 
            The hidden states are the weighted sum of the hidden states of the transformer layers with shape (B, N, D).

            Where:
                B: Batch size.
                N: Number of feature vectors.
                D: Hidden dimension of the model.
        """        
        input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

        encoder_outputs = self.encoder(
            input_features,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=None,
        )

        loss = None
        if labels is not None:
            # TODO: Implement loss calculation
            pass

        return ModelOutput(
            hidden_states=encoder_outputs,
            loss=loss
        )