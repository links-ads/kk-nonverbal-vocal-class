import torch

from non_verbal_voc_class.configs import ModelConfig
from .encoders import WhisperEncoderLayer
from .base_classifier import BaseClassifier
from transformers import WhisperForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn as nn
from typing import Optional, Tuple


class WhisperClassifier(BaseClassifier):
    def __init__(self, config: ModelConfig):
        super(WhisperClassifier, self).__init__(config)
        assert config.model_type.lower() == 'whisper', "model type must be whisper"

        self.config = config
        self.finetune_method = self.config.finetune_method

        self.model = WhisperForAudioClassification.from_pretrained(
            config.audio_model_name,
            use_weighted_layer_sum=config.use_weighted_layer_sum,
            num_labels=config.num_labels,
        )

        # Read the model config
        self.model_config = self.model.encoder.config
        if hasattr(config, 'finetune_method'):
            setattr(self.model_config, 'finetune_method', config.finetune_method)
        if hasattr(config, 'adapter_hidden_dim'):
            setattr(self.model_config, 'adapter_hidden_dim', config.adapter_hidden_dim)
        if hasattr(config, 'embedding_prompt_dim'):
            setattr(self.model_config, 'embedding_prompt_dim', config.embedding_prompt_dim)
        if hasattr(config, 'lora_rank'):
            setattr(self.model_config, 'lora_rank', config.lora_rank)

        self._add_adapter_and_freeze()

    def _add_adapter_and_freeze(self):
        state_dict = self.model.encoder.state_dict()

        # Config encoder layers with adapter, embedding prompt or lora
        self.model.encoder.layers = nn.ModuleList(
            [WhisperEncoderLayer(self.model_config) for _ in range(self.model_config.encoder_layers)]
        )

        # Load the weights back
        msg = self.model.encoder.load_state_dict(state_dict, strict=False)

        if self.finetune_method == "frozen":
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        elif self.finetune_method == "finetune":
            for param in self.model.encoder.parameters():
                param.requires_grad = True
        else:
            for name, p in self.model.encoder.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
    
    def forward(
            self,
            input_features: torch.FloatTensor,
            head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ) -> SequenceClassifierOutput:

        output = self.model(
            input_features=input_features,
            head_mask=head_mask,
            encoder_outputs=encoder_outputs,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
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