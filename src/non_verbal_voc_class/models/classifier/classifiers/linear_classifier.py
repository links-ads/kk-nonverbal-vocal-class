import torch
import torch.nn as nn

from it_multimodal_er.configs import ModelConfig
from torch import Tensor

class LinearClassifier(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.projector = nn.Linear(model_config.hidden_size, model_config.classifier_proj_size)
        self.classifier = nn.Linear(model_config.classifier_proj_size, model_config.num_labels)
    
    def forward(self, encoder_hidden_states: Tensor, length: Tensor = None) -> Tensor:
        """
        
        """
        last_hidden_states = encoder_hidden_states[-1]
        features = self.projector(last_hidden_states)
        
        # Pooling
        if length is not None:
            length = length.cuda()
            masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
            masks = masks.float()
            features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
        else:
            features = torch.mean(features, dim=1)

        logits = self.classifier(features)
        return logits