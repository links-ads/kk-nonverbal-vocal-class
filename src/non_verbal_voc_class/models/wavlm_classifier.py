import torch

from ..configs import ModelConfig
from .encoders import WavLMEncoderLayer
from .base_classifier import BaseClassifier
from transformers import WavLMForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn as nn
from typing import Optional


class WavLMClassifier(BaseClassifier):
    def __init__(self, config: ModelConfig):
        super(WavLMClassifier, self).__init__(config)
        assert config.model_type.lower() == 'wavlm', "model type must be wavlm"

        self.config = config
        self.finetune_method = self.config.finetune_method

        self.model = WavLMForSequenceClassification.from_pretrained(
            config.audio_model_name,
            use_weighted_layer_sum=config.use_weighted_layer_sum,
        )

        # Read the model config
        self.model_config = self.model.wavlm.config
        setattr(self.model_config, 'finetune_method', config.finetune_method)
        setattr(self.model_config, 'adapter_hidden_dim', config.adapter_hidden_dim)
        setattr(self.model_config, 'embedding_prompt_dim', config.embedding_prompt_dim)
        setattr(self.model_config, 'lora_rank', config.lora_rank)

        self.__add_adapter_and_freeze()

    def __add_adapter_and_freeze(self):
        state_dict = self.model.wavlm.state_dict()

        # Config encoder layers with adapter, embedding prompt or lora
        self.model.wavlm .encoder.layers = nn.ModuleList(
            [WavLMEncoderLayer(self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
        )

        # Load the weights back
        msg = self.model.wavlm.load_state_dict(state_dict, strict=False)

        if self.finetune_method == "frozen":
            for param in self.model.wavlm.parameters():
                param.requires_grad = False
        elif self.finetune_method == "finetune":
            for param in self.model.wavlm.parameters():
                param.requires_grad = True
        else:
            for name, p in self.model.wavlm.named_parameters():
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