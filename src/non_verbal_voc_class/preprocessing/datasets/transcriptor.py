import os
import librosa
import pandas as pd

from transformers import (
    pipeline, 
    AutoFeatureExtractor,
    AutoTokenizer
)
from tqdm import tqdm
import argparse

_PIPELINE_TASK = "automatic-speech-recognition"
_TOKENIZER_TASK = "transcribe"
_TOKENIZER_LANGUAGE = "italian"

class Transcriptor:
    def __init__(self, model_name, save_path, **kwargs):
        self.model_name = model_name
        self.save_path = save_path
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, language=_TOKENIZER_LANGUAGE, task=_TOKENIZER_TASK)
        self.forced_decoder_ids = self.tokenizer.get_decoder_prompt_ids(language=_TOKENIZER_LANGUAGE, task=_TOKENIZER_TASK)
        self.asr_pipeline = pipeline(_PIPELINE_TASK, model=model_name, tokenizer=self.tokenizer,**kwargs)

    def align_transcriptions_to_splits(self, transcriptions_file_path: str, split_dataframes: str) -> None:
        files_train_df = split_dataframes["train"]
        files_val_df = split_dataframes["val"]
        files_test_df = split_dataframes["test"]

        transcriptions_file_df = pd.read_csv(transcriptions_file_path)

        transcriptions_train_df = pd.DataFrame()
        transcriptions_val_df = pd.DataFrame()
        transcriptions_test_df = pd.DataFrame()

        for _, row in transcriptions_file_df.iterrows():
            file_name = row["file_name"]
            transcription = row["text"]
            # label = row["label"]

            if file_name in files_train_df["file_name"].values:
                label = files_train_df[files_train_df["file_name"] == file_name]["emotion"].values[0]
                transcriptions_train_df = pd.concat([transcriptions_train_df, pd.DataFrame([{
                    "file_name": file_name,
                    "text": transcription,
                    "label": label
                }])], ignore_index=True)
            elif file_name in files_val_df["file_name"].values:
                label = files_val_df[files_val_df["file_name"] == file_name]["emotion"].values[0]
                transcriptions_val_df = pd.concat([transcriptions_val_df, pd.DataFrame([{
                    "file_name": file_name,
                    "text": transcription,
                    "label": label
                }])], ignore_index=True)
            elif file_name in files_test_df["file_name"].values:
                label = files_test_df[files_test_df["file_name"] == file_name]["emotion"].values[0]
                transcriptions_test_df = pd.concat([transcriptions_test_df, pd.DataFrame([{
                    "file_name": file_name,
                    "text": transcription,
                    "label": label
                }])], ignore_index=True)
            else:
                raise ValueError(f"File {file_name} not found in any split.")
            
        transcriptions_train_df.to_csv(f"{self.save_path}train.csv", index=False)
        transcriptions_val_df.to_csv(f"{self.save_path}val.csv", index=False)
        transcriptions_test_df.to_csv(f"{self.save_path}test.csv", index=False)        

    def __call__(self, split_dataframes: dict):
        transcriptions_train = []
        transcriptions_val = []
        transcriptions_test = []

        for split, df in split_dataframes.items():
            for _, row in tqdm(df.iterrows(), desc=f"Transcribing {split}", total=len(df)):
                file_path = row["file_name"]
                label = row["emotion"]
                
                input_frames, _ = librosa.load(
                    file_path, 
                    sr=self.feature_extractor.sampling_rate # Load with the sampling rate of the model (16 kHz)
                ) 
                
                # Process the audio file
                transcription = self.asr_pipeline(
                    inputs={
                        "raw": input_frames,
                        "sampling_rate": self.feature_extractor.sampling_rate,
                    }, 
                    generate_kwargs={
                        "forced_decoder_ids": self.forced_decoder_ids
                    }
                )['text']

                if split == "train":
                    transcriptions_train.append([file_path, transcription, label])
                elif split == "val":
                    transcriptions_val.append([file_path, transcription, label])
                else:
                    transcriptions_test.append([file_path, transcription, label])
        
        df_train = pd.DataFrame(transcriptions_train, columns=["file_name", "text", "label"])
        df_val = pd.DataFrame(transcriptions_val, columns=["file_name", "text", "label"])
        df_test = pd.DataFrame(transcriptions_test, columns=["file_name", "text", "label"])

        df_train.to_csv(f"{self.save_path}train.csv", index=False)
        df_val.to_csv(f"{self.save_path}val.csv", index=False)
        df_test.to_csv(f"{self.save_path}test.csv", index=False)

def main():
    split_dataframes = {
        "train": pd.read_csv(args.train_csv),
        "val": pd.read_csv(args.val_csv),
        "test": pd.read_csv(args.test_csv)
    }

    transcriptor = Transcriptor(model_name=args.model_name, save_path=args.save_path)
    transcriptor.align_transcriptions_to_splits(args.transcriptions_file, split_dataframes)
    # transcriptor(split_dataframes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using a pretrained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pretrained model to use for transcription.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the transcriptions.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--transcriptions_file", type=str, required=False, help="Path to the transcriptions CSV file.")

    args = parser.parse_args()

    main()