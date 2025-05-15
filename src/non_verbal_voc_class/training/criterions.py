import numpy as np
import torch
from torch.nn import functional as F
# from src.it_peft_ser.losses import NTXentLoss, CosineEmbeddingLoss, RelaxedContrastiveLoss, CentralMomentDiscrepancyLoss

# from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss, FocalLoss, TverskyLoss, LovaszLoss

def make_loss(hparams):
    """ Loss factory: make loss from hparams """
    return CustomLoss(hparams)

class CustomLoss(torch.nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        hparams = vars(hparams)
        
        if isinstance(hparams['LOSS'], str):
            hparams['LOSS'] = [hparams['LOSS']]     # convert to list
        self.losses = self._custom_loss(hparams)    # make specified loss combination

        self.weights = hparams.get('LOSS_WEIGHTS', [1]*len(hparams['LOSS']))
        if isinstance(self.weights, (float, int)):
            self.weights = [self.weights]           # convert to list
        
        assert len(self.losses) == len(self.weights), f"The number of losses and weights must be the same. Got {len(self.losses)} losses and {len(self.weights)} weights"
        assert all([w >= 0 for w in self.weights]), f"Weight of each loss must be non-negative. Got weights: {self.weights}"


    def forward(self, logits, targets):
        loss = 0
        for l,w in zip(self.losses, self.weights):
            loss += w * l(logits, targets)
        return loss
    

    def _custom_loss(self, hparams):
        # TODO add more losses
        
        available_losses = ['cross-entropy']
        assert all([l in available_losses for l in hparams['LOSS']]), f"Available losses are: {available_losses}. Got {hparams['LOSS']}"
        
        # define custom loss function
        losses = []

        for l in hparams['LOSS']:
            
            if l == 'cross-entropy':
                label_smoothing = torch.tensor(hparams['CE_LABEL_SMOOTHING'], requires_grad=False).to(hparams['GPU_ID'])
                criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                                    reduction='mean',)
            
            losses.append(criterion)

        return losses

    def __repr__(self):
        return super().__repr__() + f" with losses: {self.losses}\nWeights: {self.weights}"