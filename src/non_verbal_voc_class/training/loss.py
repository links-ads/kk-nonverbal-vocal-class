from torch.nn import CrossEntropyLoss
from torch import Tensor

def get_loss_function(label_weights: Tensor, reduction: str = "mean") -> CrossEntropyLoss:
    """
    Function to get the loss function for the model.

    Args:
        label_weights (torch.Tensor): The weights for the labels.

    Returns:
        CrossEntropyLoss: The loss function.
    """
    return CrossEntropyLoss(
        weight=label_weights,
        reduction=reduction,
    )