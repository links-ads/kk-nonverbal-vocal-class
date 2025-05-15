import logging

from .base_preprocessor import BasePreprocessor
from ..config import PreprocessorConfig
from transformers import AutoTokenizer
from datasets import (
    load_dataset, 
    DatasetDict,
    Dataset,
)

class TextPreprocessor(BasePreprocessor):
    def __init__(self, preprocessor_config: PreprocessorConfig):
        """
        Preprocessor class for text datasets.

        args:
            - text_model_name: str, path to the huggingface model to use for feature extraction
            - max_length: int, maximum length of the text files
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(preprocessor_config.text_model_name)
        self.num_proc = preprocessor_config.num_proc
        self.label2id = preprocessor_config.label2id
        self.datasets_path = preprocessor_config.datasets_path
        self.text_dataset_path = preprocessor_config.text_dataset_path

    def preprocess(self) -> Dataset:
        """
        Preprocess the dataset.

        returns:
            - dataset: Dataset, preprocessed dataset with the following columns:
                - input_ids: List[int], tokenized text
                - labels: List[int], labels for the text
        """
        def _preprocess_function(examples):
            labels = [self.label2id[label] for label in examples["label"]]

            results = self.tokenizer(
                examples["text"], 
                padding=True, 
                truncation=True,
                return_attention_mask=True,
            )

            results["labels"] = labels
            return results

        # Load datasets
        train_dataset = load_dataset(
            "csv", 
            data_files=f"{self.datasets_path}{self.text_dataset_path}/train.csv",
            split="train"
        )
        
        val_dataset = load_dataset(
            "csv", 
            data_files=f"{self.datasets_path}{self.text_dataset_path}/val.csv",
            split="train"
        )
        
        test_dataset = load_dataset(
            "csv", 
            data_files=f"{self.datasets_path}{self.text_dataset_path}/test.csv",
            split="train"
        )
        
        # Join datasets
        dataset = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })
        
        # Preprocess datasets
        dataset = dataset.map(
            _preprocess_function, 
            batched=True, 
            num_proc=self.num_proc,
            remove_columns=["text", "label"]
        )
        
        # Set format
        dataset.set_format(
            type="torch",
            columns=[
                "input_ids", 
                "attention_mask", 
                "labels"
            ]
        )
        
        return dataset

        