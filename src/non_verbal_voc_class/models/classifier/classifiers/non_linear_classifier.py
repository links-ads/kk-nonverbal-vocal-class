import torch
import torch.nn as nn

from it_multimodal_er.configs import ModelConfig
from torch import Tensor


class NonLinearClassifier(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(model_config.hidden_size, model_config.classifier_proj_size, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(model_config.classifier_proj_size, model_config.classifier_proj_size, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(model_config.classifier_proj_size, model_config.classifier_proj_size, 1, padding=0)
        )
        
        self.out_layer = nn.Sequential(
            nn.Linear(model_config.classifier_proj_size, model_config.classifier_proj_size),
            nn.ReLU(),
            nn.Linear(model_config.classifier_proj_size, model_config.num_labels),
        )
    
    def forward(self, encoder_hidden_states: Tensor, length: Tensor =None) -> Tensor:
        """
        
        """
        features = encoder_hidden_states[-1] # last hidden state

        # Pass the weighted average to point-wise 1D Conv
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