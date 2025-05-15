"""
    Part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
    and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
"""
import torch

from .encoder_layers import Wav2Vec2EncoderLayer
from it_multimodal_er.configs import ModelConfig
from .base_audio_model import BaseAudioModel
from transformers import Wav2Vec2Model
from torch import nn as nn, Tensor
from typing import Optional


class Wav2Vec2ModelWrapper(nn.Module, BaseAudioModel):
    def __init__(self, config: ModelConfig):
        super(Wav2Vec2ModelWrapper, self).__init__()
        # 1. Load the model first with weights
        self.config = config
        self.backbone_model = Wav2Vec2Model.from_pretrained(
            config.audio_model_name,
            output_hidden_states=True,
        )
        state_dict = self.backbone_model.state_dict()

        # 2. Read the model config
        self.model_config = self.backbone_model.config
        setattr(self.model_config, 'finetune_method', config.finetune_method)
        setattr(self.model_config, 'adapter_hidden_dim', config.adapter_hidden_dim)
        setattr(self.model_config, 'embedding_prompt_dim', config.embedding_prompt_dim)
        setattr(self.model_config, 'lora_rank', config.lora_rank)
        
        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(self.model_config) for _ in range(self.model_config.num_hidden_layers)]
        )

        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        # 5. Freeze the weights
        for name, p in self.backbone_model.named_parameters():
            if name in msg.missing_keys: p.requires_grad = True
            else: p.requires_grad = False

        self.finetune_method = self.config.finetune_method
        
    # From huggingface
    def _get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        
        for kernel_size, stride in zip(self.backbone_model.config.conv_kernel, self.backbone_model.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
    
    def forward(self, input_features: Tensor, attention_mask: Optional[Tensor] = None, length: Tensor = None) -> dict[str, Tensor]:
        """
            It returns a tuple of hidden states of the transformer encoder + the input hidden states.

            Args:
            --------
                - input_features: Tensor, preprocessed (AutoFeatureExtractor) input features (batch_size, seq_len)
                - length: Tensor, length of the audio arrays

            Returns:
            --------
                - hidden_states: Tuple of hidden states of the transformer encoder + the input hidden states
                - length: Tensor, length of the audio arrays
        """
        # 1. feature extraction and projections
        with torch.no_grad():
            hidden_states = self.backbone_model.feature_extractor(input_features)
            hidden_states = hidden_states.transpose(1, 2) # New version of huggingface
            hidden_states, _ = self.backbone_model.feature_projection(hidden_states) # New version of huggingface
        
        # 2. get length and mask
        if length is not None:
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
            # length = length.cuda()
            
        # 3. transformer encoding features
        hidden_states = self.backbone_model.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states
        
        return {
            'encoder_hidden_states': hidden_states, 
            'length': length
        }