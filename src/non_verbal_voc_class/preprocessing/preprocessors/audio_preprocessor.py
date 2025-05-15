import torch
import torchaudio

from .base_preprocessor import BasePreprocessor
from ..config import PreprocessorConfig
from transformers import AutoFeatureExtractor
from pathlib import Path
from datasets import (
    Dataset,
    load_dataset,
    load_from_disk,
)

class AudioPreprocessor(BasePreprocessor):
    def __init__(self, preprocessing_config: PreprocessorConfig):
        """
        Preprocessor class for audio datasets.

        args:
            - audio_model_name: str, path to the huggingface model to use for feature extraction
            - max_duration: int, maximum duration of the audio files in seconds
        """

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(preprocessing_config.audio_model_name)
        self.label2id = preprocessing_config.label2id
        self.max_duration = preprocessing_config.max_duration
        self.target_sampling_rate = preprocessing_config.target_sampling_rate
        self.num_proc = preprocessing_config.num_proc
        self.datasets_path = preprocessing_config.datasets_path
        self.audio_dataset_path = preprocessing_config.audio_dataset_path
        self.dataset_name = preprocessing_config.dataset_name

    def preprocess(self) -> Dataset:
        """
        Preprocess the dataset.

        returns:
            - dataset: Dataset, preprocessed dataset.

        """
        # Taken from: https://colab.research.google.com/drive/1P2qHb7mwYZSPxbQZ07S7NDH-2vWv38YS?usp=sharing#scrollTo=v_BzbTu_hapW
        def speech_file_to_array_fn(path):
            speech_array, sampling_rate = torchaudio.load(path)
            
            # If there is more than 1 channel in your audio (stereo e. g. emovo)
            if speech_array.shape[0] > 1:
                # Do a mean of all channels and keep it in one channel
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            target_size = int(self.max_duration * self.target_sampling_rate)


            if len(speech) > target_size:
                return speech[:target_size]
            
            return speech

        def preprocess_function(examples):
            speech_list = [speech_file_to_array_fn(path) for path in examples["file_name"]]
            result = self.feature_extractor(
                speech_list,
                sampling_rate=self.target_sampling_rate,
                # return_attention_mask=True,
                max_length=self.max_duration * self.target_sampling_rate,
                padding="max_length",
            )
            result["labels"] = [self.label2id[label.lower()] for label in examples["emotion"]]
            result["input_features"] = result["input_values" if "input_values" in result else "input_features"]
            result.pop('input_values', None)

            return result

        # audio_dataset_preprocessed_dir = Path('data/audio_dataset_preprocessed')
        # if audio_dataset_preprocessed_dir.exists():
        #     dataset = load_from_disk("data/audio_dataset_preprocessed")
        #     return dataset
        
        dataset = load_dataset(
            path=f"{self.datasets_path}{self.audio_dataset_path}",
            name=f"default-{self.dataset_name}",
        )

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=['file_name', 'emotion']
        )

        dataset.set_format(
            type="torch",
            columns=[
                "input_features", 
                # "attention_mask",
                "labels"
            ]
        )

        # save dataset
        # dataset.save_to_disk("data/audio_dataset_preprocessed")

        return dataset