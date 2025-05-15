from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List
from torch import Tensor
import torch

class AudioCollator:
    def __init__(self, padding_value=0):
        """
        Initializes the collator with a specified padding value.

        Args:
            padding_value (int, optional): The value to use for padding. Defaults to 0.

        Returns:
            None
        """
        self.padding_value = padding_value

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        input_features = [item['input_features'] for item in batch]
        # attention_mask = [item['attention_mask'] for item in batch]
        labels = [Tensor(item['labels']) for item in batch]

        padded_input_features = pad_sequence(input_features, batch_first=True, padding_value=self.padding_value)
        # padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=self.padding_value)
        padded_labels = torch.stack(labels)

        return {
            'input_features': padded_input_features,
            # 'attention_mask': padded_attention_mask,
            'labels': padded_labels
        }