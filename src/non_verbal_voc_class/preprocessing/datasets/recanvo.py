import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def _create_subsamples(data_path, new_folder):
    files = os.listdir(data_path)
    df_list = []
    metadata = pd.read_csv(f"{data_path}dataset_file_directory.csv")

    for file in files:
        if ".wav" not in file:
            continue

        file_metadata = metadata[metadata['Filename'] == file]
        
        if file_metadata.empty:
            continue

        # copy file to the new folder
        shutil.copy(
            src=f"{data_path}{file}", 
            dst=f"{data_path}{new_folder}/{file}"
        )

        file = {
            'file_name': file,
            'speaker_id': file_metadata['Participant'].values[0],
            'label': file_metadata['Label'].values[0],
        }

        df_list.append(file)
    
    samples_df = pd.DataFrame(df_list)
    
    return samples_df


def _split_dataset(samples_df, train_size, test_size):
    """
    Split the dataset into train, val, and test DataFrames.

    args:
        - samples_df: pandas DataFrame, containing the metadata
        - train_size: float, size of the train set
        - test_size: float, size of the test set

    return:
        - train_df: pandas DataFrame, containing the train set
        - val_df: pandas DataFrame, containing the val set
        - test_df: pandas DataFrame, containing the test set
    """
    train_df, test_df = train_test_split(
        samples_df, 
        train_size=train_size, 
        stratify=samples_df['label'], 
        random_state=42
    )

    # Calculate and print number of samples per class
    print("Number of samples per class:")
    class_counts = samples_df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

    train_counts = train_df['label'].value_counts()
    print("\nTrain set distribution:")
    for label, count in train_counts.items():
        print(f"  {label}: {count} ({count/len(train_df):.2%})")

    test_counts = test_df['label'].value_counts()
    print("\nTest set distribution:")
    for label, count in test_counts.items():
        print(f"  {label}: {count} ({count/len(test_df):.2%})")

    val_df, test_df = train_test_split(
        test_df, 
        test_size=test_size, 
        stratify=test_df['label'], 
        random_state=42
    )

    return train_df, val_df, test_df


def _filter_small_classes(df, min_samples=10):
    """
    Remove classes that have fewer than the specified minimum number of samples.
    
    args:
        - df: pandas DataFrame with data samples
        - min_samples: minimum number of samples required per class
        
    return:
        - filtered_df: DataFrame with only classes that have enough samples
    """
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    filtered_df = df[df['label'].isin(valid_classes)]
    
    removed_classes = set(df['label'].unique()) - set(valid_classes)
    if removed_classes:
        print(f"Removed {len(removed_classes)} classes with fewer than {min_samples} samples:")
        for cls in removed_classes:
            count = class_counts.get(cls, 0)
            print(f"  {cls}: {count} samples")
    
    return filtered_df


def _drop_columns(df):
    """
    Drop unnecessary columns from the DataFrame.

    args:
        - df: pandas DataFrame

    return:
        - df: pandas DataFrame, with unnecessary columns dropped
    """
    df = df.drop(columns=['speaker_id'])

    return df


def _shuffle_dataset(df):
    """
    Shuffle the rows of the DataFrame.

    args:
        - df: pandas DataFrame

    return:
        - df: pandas DataFrame, with shuffled rows
    """
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def _append_path(df, column_name, root_path):
    """
    Append the root path to a column in the DataFrame.

    args:
        - df: pandas DataFrame
        - column_name: str, name of the column
        - root_path: str, root path to be appended

    return:
        - df: pandas DataFrame, with the root path appended to the column
    """
    df[column_name] = df[column_name].apply(lambda x: f"{root_path}{x}")

    return df


def _save_dataset(df, file_path):
    """
    Save the DataFrame as a CSV file.

    args:
        - df: pandas DataFrame
        - file_path: str, path to save the file

    return:
        - None
    """
    df.to_csv(file_path, index=False)

def _create_readme(samples_path):
    with open(f"{samples_path}README.md", "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default-recanvo\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write('    path: "train.csv"\n')
        f.write("  - split: val\n")
        f.write('    path: "val.csv"\n')
        f.write("  - split: test\n")
        f.write('    path: "test.csv"\n')
        f.write("---\n")


def prepare_recanvo(data_path="data/ReCANVo/", train_size=0.8, test_size=0.5, min_samples_per_class=10):
    """
    Prepare the ReCANVo dataset. Creates train, val, test splits and saves them into the dataset folder.

    args:
        - data_path: str, path to the ReCANVo dataset
        - train_size: float, size of the train set
        - test_size: float, size of the test set
        - min_samples_per_class: int, minimum samples required per class

    return:
        - samples_path: str, path to the samples directory
    """
    # Ensure data path ends with a slash
    if not data_path.endswith('/'):
        data_path += '/'
        
    samples_path = f"{data_path}samples/"

    # Create samples folder
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    # Create metadata
    samples_df = _create_subsamples(data_path, "samples")
    
    # Filter out classes with too few samples
    if min_samples_per_class > 1:
        samples_df = _filter_small_classes(samples_df, min_samples_per_class)
        print(f"After filtering, {len(samples_df)} samples remain across {len(samples_df['label'].unique())} classes")

    # Split the dataset
    train_df, val_df, test_df = _split_dataset(samples_df, train_size, test_size)

    # Filter out classes with fewer than specified minimum samples
    train_df = _filter_small_classes(train_df, min_samples=min_samples_per_class)
    val_df = _filter_small_classes(val_df, min_samples=min_samples_per_class)
    test_df = _filter_small_classes(test_df, min_samples=min_samples_per_class)

    # Drop unnecessary columns
    train_df = _drop_columns(train_df)
    val_df = _drop_columns(val_df)
    test_df = _drop_columns(test_df)

    # Shuffle the datasets
    train_df = _shuffle_dataset(train_df)
    val_df = _shuffle_dataset(val_df)
    test_df = _shuffle_dataset(test_df)

    # Append full path to file_name column
    train_df = _append_path(train_df, 'file_name', samples_path)
    val_df = _append_path(val_df, 'file_name', samples_path)
    test_df = _append_path(test_df, 'file_name', samples_path)

    # Save the split datasets into the folder
    _save_dataset(train_df, f"{samples_path}train.csv")
    _save_dataset(val_df, f"{samples_path}val.csv")
    _save_dataset(test_df, f"{samples_path}test.csv")

    # Create README file
    _create_readme(samples_path)

    return samples_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ReCANVo dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ReCANVo/",
        help="Path to the ReCANVo dataset (default: data/ReCANVo/)"
    )
    parser.add_argument(
        "--train-size", 
        type=float, 
        default=0.8,
        help="Size of the training set (default: 0.8)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        help="Size of the test set from non-training data (default: 0.5)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per class (default: 10)"
    )
    args = parser.parse_args()

    print(f"Preparing ReCANVo dataset at {args.data_path}")
    print(f"Parameters: train_size={args.train_size}, test_size={args.test_size}, min_samples={args.min_samples}")
    
    samples_path = prepare_recanvo(
        data_path=args.data_path,
        train_size=args.train_size,
        test_size=args.test_size,
        min_samples_per_class=args.min_samples
    )
    
    print(f"Dataset prepared and saved to {samples_path}")
    print(f"Train, validation, and test splits created")