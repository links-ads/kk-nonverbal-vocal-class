import numpy as np

from transformers import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation.
    
    Args:
    - eval_pred: EvalPrediction object compose of 'predictions', 'labels_ids', 'inputs' (if set to return inputs in Trainer).

    it returns a dictionary of metrics.
    """
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    weighted_accuracy = balanced_accuracy_score(labels, predictions)
    
    precision_macro = precision_score(labels, predictions, average='macro')
    precision_weighted = precision_score(labels, predictions, average='weighted')
    
    recall_macro = recall_score(labels, predictions, average='macro')
    recall_weighted = recall_score(labels, predictions, average='weighted')

    f1_score_macro = f1_score(labels, predictions, average='macro')
    f1_score_weighted = f1_score(labels, predictions, average='weighted')

    report = {
        'accuracy': round(accuracy*100, 2),
        'weighted_accuracy': round(weighted_accuracy*100, 2),
        'precision_macro': round(precision_macro*100, 2),
        'precision_weighted': round(precision_weighted*100, 2),
        'recall_macro': round(recall_macro*100, 2),
        'recall_weighted': round(recall_weighted*100, 2),
        'f1_score_macro': round(f1_score_macro*100, 2),
        'f1_score_weighted': round(f1_score_weighted*100, 2),
    }

    return report