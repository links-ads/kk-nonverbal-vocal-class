import os
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def _parse_filename(filename, class_label):
    """
    Parse donateacry filename to extract metadata.
    Format: app_instance_uuid-unix_timestamp-app_version-gender-age-reason.wav
    """
    if not filename.endswith('.wav'):
        return None
        
    name_without_ext = filename.replace('.wav', '')
    parts = name_without_ext.split('-')
    
    if len(parts) < 6:
        return None
    
    app_instance_uuid = parts[0]
    timestamp = parts[1]
    app_version = parts[2]
    gender = parts[-3]  # m for male, f for female
    age = parts[-2]     # age range (e.g., 04 for 0-4 weeks)
    reason = parts[-1]  # reason for crying (e.g., hu for hunger)
    
    return {
        'label': class_label,
        'app_instance_uuid': app_instance_uuid,
        'timestamp': timestamp,
        'app_version': app_version,
        'gender': gender,
        'age': age,
        'reason': reason
    }


def _create_subsamples(data_path, new_folder):
    """
    Create samples by copying audio files to a new folder and extracting metadata.
    """
    df_list = []
    
    samples_path = f"{data_path}{new_folder}/"
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        
        if not os.path.isdir(class_path):
            continue

        # Skip the samples folder to avoid reading it as a class
        if class_folder == "samples":
            continue
            
        for file in os.listdir(class_path):
            if not file.endswith('.wav'):
                continue
                
            file_info = _parse_filename(file, class_folder)
            if not file_info:
                continue
                
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(samples_path, file)
            
            if not os.path.exists(dst_path) or not os.path.samefile(src_path, dst_path):
                shutil.copy(src=src_path, dst=dst_path)
            
            file_metadata = {
                'file_name': file,
                'speaker_id': file_info['app_instance_uuid'],
                'label': file_info['label'],
                'gender': file_info['gender'],
                'age': file_info['age'],
                'reason': file_info['reason'],
                'timestamp': file_info['timestamp']
            }
            
            df_list.append(file_metadata)
    
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

    print("Number of samples per class:")
    class_counts = samples_df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

    train_counts = train_df['label'].value_counts()
    print("\nTrain set distribution:")
    for label, count in train_counts.items():
        print(f"  {label}: {count} ({count/len(train_df):.2%})")

    val_counts = test_df['label'].value_counts()
    print("\nValidation set distribution:")
    for label, count in val_counts.items():
        print(f"  {label}: {count} ({count/len(test_df):.2%})")

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
    """
    df = df.drop(columns=['speaker_id', 'gender', 'age', 'reason', 'timestamp'])

    return df


def _shuffle_dataset(df):
    """
    Shuffle the rows of the DataFrame.
    """
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def _append_path(df, column_name, root_path):
    """
    Append the root path to a column in the DataFrame.
    """
    df[column_name] = df[column_name].apply(lambda x: f"{root_path}{x}")

    return df


def _save_dataset(df, file_path):
    """
    Save the DataFrame as a CSV file.
    """
    df.to_csv(file_path, index=False)


def _create_readme(samples_path):
    with open(f"{samples_path}README.md", "w") as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default-donateacry\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write('    path: "train.csv"\n')
        f.write("  - split: val\n")
        f.write('    path: "val.csv"\n')
        f.write("  - split: test\n")
        f.write('    path: "test.csv"\n')
        f.write("---\n")


def prepare_donateacry(data_path="data/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/", train_size=0.8, test_size=0.5, min_samples_per_class=10):
    """
    Prepare the donateacry dataset. Creates train, val, test splits and saves them into the dataset folder.

    args:
        - data_path: str, path to the donateacry dataset
        - train_size: float, size of the train set
        - test_size: float, size of the test set
        - min_samples_per_class: int, minimum samples required per class

    return:
        - samples_path: str, path to the samples directory
    """
    if not data_path.endswith('/'):
        data_path += '/'
        
    samples_path = f"{data_path}samples/"

    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    samples_df = _create_subsamples(data_path, "samples")
    
    if min_samples_per_class > 1:
        samples_df = _filter_small_classes(samples_df, min_samples_per_class)
        print(f"After filtering, {len(samples_df)} samples remain across {len(samples_df['label'].unique())} classes")

    train_df, val_df, test_df = _split_dataset(samples_df, train_size, test_size)

    # train_df = _filter_small_classes(train_df, min_samples=min_samples_per_class)
    # val_df = _filter_small_classes(val_df, min_samples=min_samples_per_class)
    # test_df = _filter_small_classes(test_df, min_samples=min_samples_per_class)

    train_df = _drop_columns(train_df)
    val_df = _drop_columns(val_df)
    test_df = _drop_columns(test_df)

    train_df = _shuffle_dataset(train_df)
    val_df = _shuffle_dataset(val_df)
    test_df = _shuffle_dataset(test_df)

    train_df = _append_path(train_df, 'file_name', samples_path)
    val_df = _append_path(val_df, 'file_name', samples_path)
    test_df = _append_path(test_df, 'file_name', samples_path)

    _save_dataset(train_df, f"{samples_path}train.csv")
    _save_dataset(val_df, f"{samples_path}val.csv")
    _save_dataset(test_df, f"{samples_path}test.csv")

    _create_readme(samples_path)

    return samples_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare donateacry dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/",
        help="Path to the donateacry dataset (default: data/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/)"
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

    print(f"Preparing donateacry dataset at {args.data_path}")
    print(f"Parameters: train_size={args.train_size}, test_size={args.test_size}, min_samples={args.min_samples}")
    
    samples_path = prepare_donateacry(
        data_path=args.data_path,
        train_size=args.train_size,
        test_size=args.test_size,
        min_samples_per_class=args.min_samples
    )
    
    print(f"Dataset prepared and saved to {samples_path}")
    print(f"Train, validation, and test splits created")