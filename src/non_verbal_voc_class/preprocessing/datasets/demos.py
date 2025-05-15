import os
import torch
import argparse
import pandas as pd

from .stratified_group_shuffle_split import stratified_group_shuffle_split

def _create_metadata(datasets_path):
    files = os.listdir(f"{datasets_path}multimodal_demos/")

    df_list = []
    for file in files:
        if ".wav" not in file:
            continue

        features = file.split('.')[0]

        sentence_id = features[-6:]
        emotion = features[-6:-3]

        file = {
            'file_name': file,
            'sentence_id': sentence_id,
            'emotion': emotion
        }

        df_list.append(file)

    samples_df = pd.DataFrame(df_list)
    return samples_df

def _append_file_path(df, datasets_path):
    df['file_name'] = df['file_name'].apply(lambda x: f"{datasets_path}multimodal_demos/{x}")
    return df

def _split_dataset(samples_df, train_size):
    # Ensure same sentences are not in the same split
    all_labels_present_in_splits = False

    while not all_labels_present_in_splits:
        train_df, val_df, test_df = stratified_group_shuffle_split(
                df_main=samples_df,
                group_column="sentence_id",
                label_column="emotion",
                train_proportion=train_size
            )    
        
        if train_df.emotion.nunique() == val_df.emotion.nunique() == test_df.emotion.nunique():
            all_labels_present_in_splits = True

    return train_df, val_df, test_df

def _save_datasets(train_df, val_df, test_df, datasets_path):
    train_df.to_csv(f"{datasets_path}multimodal_demos/train.csv", index=False)
    val_df.to_csv(f"{datasets_path}multimodal_demos/val.csv", index=False)
    test_df.to_csv(f"{datasets_path}multimodal_demos/test.csv", index=False)

def _create_readme(datasets_path):
    with open(f"{datasets_path}multimodal_demos/README.md", "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default-demos\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write('    path: "train.csv"\n')
        f.write("  - split: val\n")
        f.write('    path: "val.csv"\n')
        f.write("  - split: test\n")
        f.write('    path: "test.csv"\n')
        f.write("---\n")

def prepare_demos(
        datasets_path: str,
        train_size: float
    ):

    # Create metadata
    samples_df = _create_metadata(datasets_path)

    # Append file path
    samples_df = _append_file_path(samples_df, datasets_path)

    # Split dataset
    train_df, val_df, test_df = _split_dataset(samples_df, train_size)

    # Save datasets
    _save_datasets(train_df, val_df, test_df, datasets_path)

    # Create README file
    _create_readme(datasets_path)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the DEMoS dataset.")
    parser.add_argument("--datasets_path", type=str, required=True, help="Path to the datasets directory")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of the dataset to include in the train split")
    args = parser.parse_args()

    datasets_path = args.datasets_path
    train_size = args.train_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_demos(
        datasets_path=datasets_path,
        train_size=train_size,
        device=device
    )
