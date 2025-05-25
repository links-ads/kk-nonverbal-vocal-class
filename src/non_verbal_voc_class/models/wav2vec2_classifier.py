"""
    Part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
    and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
"""
import torch

from ..configs import ModelConfig
from .encoders import Wav2Vec2EncoderLayer
from .base_classifier import BaseClassifier
from transformers import Wav2Vec2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn as nn
from typing import Optional, Union, Tuple


class Wav2Vec2Classifier(BaseClassifier):
    def __init__(self, config: ModelConfig):
        super(Wav2Vec2Classifier, self).__init__(config)
        assert config.model_type.lower() == 'wav2vec2', "model type must be wav2vec2"

        self.config = config
        self.finetune_method = self.config.finetune_method

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            config.audio_model_name,
        )

        # Read the model config
        self.model_config = self.model.wav2vec2.config
        setattr(self.model_config, 'finetune_method', config.finetune_method)
        setattr(self.model_config, 'adapter_hidden_dim', config.adapter_hidden_dim)
        setattr(self.model_config, 'embedding_prompt_dim', config.embedding_prompt_dim)
        setattr(self.model_config, 'lora_rank', config.lora_rank)

        self.__add_adapter_and_freeze()

    def __add_adapter_and_freeze(self):
        state_dict = self.model.wav2vec2.state_dict()

        # Config encoder layers with adapter, embedding prompt or lora
        self.model.wav2vec2.encoder.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(self.model_config) for _ in range(self.model_config.num_hidden_layers)]
        )

        # Load the weights back
        msg = self.model.wav2vec2.load_state_dict(state_dict, strict=False)

        if self.finetune_method == "frozen":
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = False
        elif self.finetune_method == "finetune":
            for param in self.model.wav2vec2.parameters():
                param.requires_grad = True
        else:
            for name, p in self.model.wav2vec2.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
    
    def forward(
            self,
            input_features: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> SequenceClassifierOutput:

        output = self.model(
            input_values=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
            labels=None,
        )
        logits = output.logits
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )