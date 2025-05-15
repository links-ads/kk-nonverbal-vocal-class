from transformers import Trainer
from pathlib import Path

import csv
import os

def eval_trainer(
        trainer: Trainer,
        dataset: dict,
        evaluate_column: str,
        backbone_name: str,
        dataset_name: str,
        classifier_name: str,
        results_file: str
):
    
    results = trainer.evaluate(dataset[evaluate_column])
    results = {k.replace("eval_", ""):v for k, v in results.items()}
    results = {k:round(v,4) for k, v in results.items() if k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_score_macro']}
    
    os.makedirs('results', exist_ok=True)

    # Create the CSV file if it doesn't exist and write the column names
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['backbone', 'dataset', 'classifier_name', *list(results.keys())])

    # Open the CSV file in append mode and write the values
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([backbone_name, dataset_name, classifier_name, *list(results.values())])
