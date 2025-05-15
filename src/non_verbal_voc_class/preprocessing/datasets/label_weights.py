import pandas as pd

from datasets import Dataset
from torch import Tensor

def get_label_weights(dataset: Dataset) -> Tensor:
    """
    Return training weights. The weights are calculated as the inverse of the frequency of each class, normalized by the sum of all weights.
    This code was taken from the following source:
    https://stackoverflow.com/questions/73145394/how-can-i-take-the-unique-rows-of-a-huggingface-dataset

    :param dataset: Dataset, dataset to calculate the weights for.

    :return weights: Tensor, class weights.
    """
    dataset_df = pd.DataFrame(dataset['train']['labels'])
    value_counts = dataset_df.value_counts().sort_index()

    weights = Tensor([count for count in value_counts])
    weights = weights.sum() / weights
    weights = weights / weights.sum()

    return weights