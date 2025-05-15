import torch
import torch.nn as nn

from transformers.modeling_outputs import SequenceClassifierOutput
from .whisper import WhisperWrapper
from .wav2vec import Wav2VecWrapper
from .wavlm import WavLMWrapper
from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss
from .classifiers import (
    LinearDownstreamModel,
    NonLinearDownstreamModel,
    MultiLevelDownstreamModel,
)

class CustomModelForAudioClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        args:
        - config is a WhisperConfig object where some parameters are:
        --- model_config.hidden_size: the hidden size of the upstream model (e.g., 768 for Whisper-small)
        --- model_config.num_hidden_layers: number of hidden layers in the upstream model (e.g., 12 for Whisper-small)

        --- config.output_hidden_states: whether the upstream model returns all hidden states (False by default). Set True
        --- config.classifier_proj_size: the hidden size of the downstream classification model (by default 256 as in PEFT-SER https://arxiv.org/pdf/2306.05350)
        --- config.num_labels: the number of labels for classification (2 by default). Set 4

        """
        self.config = config
        assert config.output_hidden_states == True, "The upstream model must return all hidden states"
        
        # Load upstream model with weights already frozen depending on the finetuning method (finetune, adpater, lora)
        if config.model_type == 'wav2vec2':
            self.encoder = Wav2VecWrapper(config)
        elif config.model_type == 'wavlm':
            self.encoder = WavLMWrapper(config)
        elif config.model_type == 'whisper':
            self.encoder = WhisperWrapper(config)
        else:
            raise ValueError(f"Model {config.model_type} not available. Available models: wav2vec2, wavlm, whisper")

        if config.classifier_name  == 'linear':
            self.classifier = LinearDownstreamModel(config)
        elif config.classifier_name == 'nonlinear':
            self.classifier = NonLinearDownstreamModel(config)
        elif config.classifier_name == 'multilevel':
            self.classifier = MultiLevelDownstreamModel(config, use_conv_output=True) # use the input embeddings and transformer hidden states by default
        else:
            raise ValueError(f"Classifier {config.classifier_name} not available. Available classifiers: linear, nonlinear, multilevel")
        
    def forward(
        self,
        input_features: Optional[torch.LongTensor],
        # length: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        >>> model = CustomWhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

        >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        >>> sample = next(iter(ds))

        >>> inputs = feature_extractor(
        ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
        ... )
        >>> input_features = inputs.input_features

        >>> with torch.no_grad():
        ...     logits = model(input_features).logits

        >>> predicted_class_ids = torch.argmax(logits).item()
        >>> predicted_label = model.config.id2label[predicted_class_ids]
        >>> predicted_label
        'Afrikaans'
        ```"""

        # Encoder (set the config.output_hidden_states=True)
        if encoder_outputs is None:
            encoder_output = self.encoder(
                input_features,
                # length=length,
            )

        logits = self.classifier(**encoder_output)

        loss = None

        if labels is not None:
            if self.config.lossname == 'cross-entropy':
                label_weights = torch.tensor(self.config.label_weights, device=logits.device)
                loss_fct = CrossEntropyLoss(
                    weight=label_weights,
                    reduction='mean',
                )
                # move labels to correct device to enable PP
                labels = labels.to(logits.device)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                raise ValueError(f"Loss {self.config.lossname} not available. Available losses: cross-entropy")
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )