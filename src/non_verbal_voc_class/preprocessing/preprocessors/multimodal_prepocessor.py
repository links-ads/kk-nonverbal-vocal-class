from .base_preprocessor import BasePreprocessor
from .audio_preprocessor import AudioPreprocessor
from .text_preprocessor import TextPreprocessor
from ..config import PreprocessorConfig
from datasets import Dataset, DatasetDict

class MultiModalPreprocessor(BasePreprocessor):
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.audio_preprocessor = AudioPreprocessor(config)
        self.text_preprocessor = TextPreprocessor(config)

    def preprocess(self) -> Dataset:
        # Process audio and text separately
        audio_dataset = self.audio_preprocessor.preprocess()
        text_dataset = self.text_preprocessor.preprocess()

        # Merge datasets based on file_name for each split
        merged_datasets = {}
        for split in ['train', 'val', 'test']:
            merged_datasets[split] = Dataset.from_dict({
                'input_features': audio_dataset[split]['input_features'],
                'input_ids': text_dataset[split]['input_ids'],
                # 'audio_attention_mask': audio_dataset[split]['attention_mask'],
                'text_attention_mask': text_dataset[split]['attention_mask'],
                'labels': audio_dataset[split]['labels']  # Assuming labels are the same in both datasets
            })

        # Merge datasets
        merged_dataset = DatasetDict({
            'train': merged_datasets['train'],
            'val': merged_datasets['val'],
            'test': merged_datasets['test']
        })


        # Set dataset format
        merged_dataset.set_format(
            type="torch",
            columns=[
                "input_features",
                "input_ids",
                # "audio_attention_mask",
                "text_attention_mask",
                "labels"
            ],
        )

        return merged_dataset