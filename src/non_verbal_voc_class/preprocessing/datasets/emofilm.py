import os
import logging
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/preprocessing/emofilm_dataloader.log", encoding="utf-8", level=logging.DEBUG)

def _create_metadata(datasets_path):
    emotion_mapping = {
        'rab': 'anger',
        'dis': 'disgust',
        'gio': 'joy',
        'tri': 'sadness',
        'ans': 'anxiety',
    }

    files = os.listdir(f"{datasets_path}emofilm/")

    df_list = []
    for file in files:
        if ".wav" not in file:
            continue

        features = file.split('.')[0]

        speaker_id = features[5:9]
        emotion = features[2:5]
        emotion = emotion_mapping[emotion]

        file = {
            'file_name': file,
            'speaker_id': speaker_id,
            'emotion': emotion
        }

        df_list.append(file)

    samples_df = pd.DataFrame(df_list)
    return samples_df

def _append_file_path(df, datasets_path):
    df['file_name'] = df['file_name'].apply(lambda x: f"{datasets_path}emofilm/{x}")
    return df

def _split_dataset(samples_df, train_size, test_size):
    # Ensure same sentences are not in the same split
    unique_sentences = samples_df['speaker_id'].unique()
    train_sentences, test_val_sentences = train_test_split(unique_sentences, train_size=train_size)
    val_sentences, test_sentences = train_test_split(test_val_sentences, test_size=test_size)

    train_df = samples_df[samples_df['speaker_id'].isin(train_sentences)]
    val_df = samples_df[samples_df['speaker_id'].isin(val_sentences)]
    test_df = samples_df[samples_df['speaker_id'].isin(test_sentences)]

    return train_df, val_df, test_df

def _save_datasets(train_df, val_df, test_df, datasets_path):
    train_df.to_csv(f"{datasets_path}emofilm/train.csv", index=False)
    val_df.to_csv(f"{datasets_path}emofilm/val.csv", index=False)
    test_df.to_csv(f"{datasets_path}emofilm/test.csv", index=False)

def _create_readme(datasets_path):
    with open(f"{datasets_path}emofilm/README.md", "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default-emofilm\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write('    path: "train.csv"\n')
        f.write("  - split: val\n")
        f.write('    path: "val.csv"\n')
        f.write("  - split: test\n")
        f.write('    path: "test.csv"\n')
        f.write("---\n")

def prepare_emofilm(
        datasets_path: str,
        train_size: float,
        test_size: float,
    ):
    logger.info(f"Preparing the Emofilm dataset...")

    # Create metadata
    samples_df = _create_metadata(datasets_path)

    # Append file path
    samples_df = _append_file_path(samples_df, datasets_path)

    # Split dataset
    train_df, val_df, test_df = _split_dataset(samples_df, train_size, test_size)

    # Save datasets
    _save_datasets(train_df, val_df, test_df, datasets_path)

    # Create README file
    _create_readme(datasets_path)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the DEMoS dataset.")
    parser.add_argument("--datasets_path", type=str, required=True, help="Path to the datasets directory")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the train split")
    parser.add_argument("--test_size", type=float, default=0.5, help="Proportion of the test/val split to include in the test split")
    args = parser.parse_args()

    datasets_path = args.datasets_path
    train_size = args.train_size
    test_size = args.test_size

    prepare_emofilm(
        datasets_path=datasets_path,
        train_size=train_size,
        test_size=test_size,
    )