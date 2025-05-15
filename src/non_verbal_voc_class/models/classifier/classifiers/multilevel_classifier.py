from typing import Optional

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch import Tensor

class MultiLevelClassifier(nn.Module):
    def __init__(self, model_config, use_conv_output: Optional[bool] = True) -> None:
        """
            Args:
            ------
                - model_config is the config of the upstream model (e.g., https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperConfig)
                - use_conv_output is a boolean indicating whether to use the output of the convolutional layers of the upstream model

            Useful attribute of model_config:
            ---------------------------------
                - model_config.hidden_size: the hidden size of the upstream model (FIXED)
                - model_config.num_hidden_layers: number of hidden layers in the upstream model (FIXED)
                - model_config.output_hidden_states: whether the upstream model returns all hidden states (False by default)
                - model_config.classifier_proj_size: the hidden size of the downstream classification model (by default 256 as in PEFT-SER https://arxiv.org/pdf/2306.05350)
                - model_config.num_labels: the number of labels for classification (2 by default)
        """
        super().__init__()

        self.model_config = model_config
        self.use_conv_output = use_conv_output

        self.model_seq = nn.Sequential(
            nn.Conv1d(self.model_config.hidden_size, self.model_config.classifier_proj_size, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(self.model_config.classifier_proj_size, self.model_config.classifier_proj_size, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(self.model_config.classifier_proj_size, self.model_config.classifier_proj_size, 1, padding=0)
        )

        if self.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
            self.weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))
        
        self.out_layer = nn.Sequential(
            nn.Linear(self.model_config.classifier_proj_size, self.model_config.classifier_proj_size),
            nn.ReLU(),
            nn.Linear(self.model_config.classifier_proj_size, self.model_config.num_labels),
        )

    def forward(self, encoder_hidden_states: Tensor, length: Tensor = None) -> Tensor:
        """
            The first element is the output of the convolutional layers and the rest are the hidden states of the transformer encoder.
            Each element has shape (B, T, D) where B is the batch size, T is the sequence length, and D is the hidden size (hidden_size).

            Args:
            ------
                - encoder_hidden_states is a Tuple of hidden states of the transformer encoder (number of hidden layers + 1 if self.use_conv_output else number of hidden layers).
        """

        # 1. stacked feature
        if self.use_conv_output:
            stacked_feature = torch.stack(encoder_hidden_states, dim=0)
        else:
            stacked_feature = torch.stack(encoder_hidden_states, dim=0)[1:] # exclude the convolution output
        
        # 2. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(self.model_config.num_hidden_layers + 1, -1)
        else:
            stacked_feature = stacked_feature.view(self.model_config.config.num_hidden_layers, -1)
        
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 3. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # 4. Pooling
        if length is not None:
            length = length.cuda()
            masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)
        
        # 5. Output predictions
        # B x D
        predicted = self.out_layer(features)
        return predicted # logits