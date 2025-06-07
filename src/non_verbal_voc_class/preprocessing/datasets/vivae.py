"""
VIVAE Dataset Preprocessing Script

This script preprocesses the VIVAE (Vienna Vocal Vocalize Vocalizing Adults and Emotional) dataset.
The VIVAE dataset contains emotional vocal expressions and vocalizations.

Filename format: Speaker_Emotion_Intensity_Item-ID.wav
- Speaker: Speaker ID (S01, S02, etc.)
- Emotion: Emotion type (achievement, anger, fear, pain, pleasure, surprise)
- Intensity: Intensity level (low, moderate, peak, strong)
- Item-ID: Item identifier (01, 02, etc.)

Example: S01_achievement_low_01.wav

Dataset paper: https://doi.org/10.1371/journal.pone.0252336
"""

import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple
from pathlib import Path


RANDOM_STATE = 42
DATASET_NAME = "vivae"

EMOTION_MAPPING = {
    "achievement": "achievement",
    "anger": "anger", 
    "fear": "fear",
    "pain": "pain",
    "pleasure": "pleasure",
    "surprise": "surprise"
}

INTENSITY_MAPPING = {
    "low": "low",
    "moderate": "moderate", 
    "peak": "peak",
    "strong": "strong"
}


def _parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse VIVAE filename to extract metadata.
    
    Args:
        filename: Filename in format Speaker_Emotion_Intensity_Item-ID.wav
        
    Returns:
        Dictionary with extracted metadata or None if parsing fails
    """
    try:
        name_without_ext = filename.replace('.wav', '')
        parts = name_without_ext.split('_')
        
        if len(parts) != 4:
            return None
            
        speaker, emotion, intensity, item_id = parts
        
        if emotion not in EMOTION_MAPPING or intensity not in INTENSITY_MAPPING:
            return None
            
        return {
            'file_name': filename,
            'speaker': speaker,
            'label': EMOTION_MAPPING[emotion]
        }
    except Exception:
        return None

def _split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.7,
    test_size: float = 0.15,
    stratify_col: str = 'label'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification.
    
    Args:
        df: Input dataframe
        train_size: Training set proportion
        test_size: Test set proportion
        stratify_col: Column to stratify on
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    val_size = 1.0 - train_size - test_size
    
    X = df.drop(columns=[stratify_col])
    y = df[stratify_col]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), 
        stratify=y, random_state=RANDOM_STATE
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(test_size / (val_size + test_size)),
        stratify=y_temp, random_state=RANDOM_STATE
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df


def _filter_small_classes(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """Filter out classes with fewer than min_samples."""
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    return df[df['label'].isin(valid_classes)].copy()


def _shuffle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle dataset and reset index."""
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _append_path(df: pd.DataFrame, processed_path: Path) -> pd.DataFrame:
    """Add full file paths to the dataframe."""
    df_copy = df.copy()
    df_copy['file_name'] = df_copy['file_name'].apply(
        lambda x: str(processed_path / x)
    )
    return df_copy


def _save_dataset(
    df: pd.DataFrame, 
    output_path: Path, 
    split_name: str
) -> None:
    """Save dataset split to CSV file."""
    output_file = output_path / f"{split_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {split_name} set: {len(df)} samples to {output_file}")


def _create_readme(output_path: Path) -> None:
    """Create README file with dataset statistics."""
    readme_file = output_path / "README.md"
    with open(readme_file, 'w') as f:
        f.write("---\n")
        f.write("configs:\n")
        f.write("- config_name: default-vivae\n")
        f.write("  data_files:\n")
        f.write("  - split: train\n")
        f.write('    path: "train.csv"\n')
        f.write("  - split: val\n")
        f.write('    path: "val.csv"\n')
        f.write("  - split: test\n")
        f.write('    path: "test.csv"\n')
        f.write("---\n")
        f.write("\n")


def preprocess_vivae(
    data_path: str,
    train_size: float = 0.7,
    test_size: float = 0.15,
    min_samples: int = 10
) -> None:
    """
    Preprocess VIVAE dataset.
    
    Args:
        data_path: Path to VIVAE dataset directory
        train_size: Training set proportion
        test_size: Test set proportion
        min_samples: Minimum samples per class
    """
    data_path = Path(data_path)
    audio_dir = data_path / "VIVAE" / "full_set"
    output_path = data_path / "samples"
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    print(f"Processing VIVAE dataset from {audio_dir}")
    print(f"Output directory: {output_path}")
    
    data_rows = []
    processed_files = 0
    
    for audio_file in audio_dir.glob("*.wav"):
        metadata = _parse_filename(audio_file.name)
        if metadata is not None:
            data_rows.append(metadata)

            dest_file = output_path / audio_file.name
            if not dest_file.exists():
                shutil.copy2(audio_file, dest_file)
            processed_files += 1
    
    if not data_rows:
        raise ValueError("No valid audio files found")
    
    df = pd.DataFrame(data_rows)
    print(f"Processed {processed_files} files")
    print(f"Original dataset shape: {df.shape}")
    
    df = _filter_small_classes(df, min_samples)
    print(f"After filtering small classes: {df.shape}")
    
    df = _shuffle_dataset(df)
    
    train_df, val_df, test_df = _split_dataset(df, train_size, test_size)

    train_df = _append_path(train_df, output_path)
    val_df = _append_path(val_df, output_path)
    test_df = _append_path(test_df, output_path)

    _save_dataset(train_df, output_path, "train")
    _save_dataset(val_df, output_path, "val")
    _save_dataset(test_df, output_path, "test")

    class_distribution = df['label'].value_counts().to_string()

    stats = {
        'total_files': processed_files,
        'total_classes': df['label'].nunique(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': class_distribution,
        'min_samples': min_samples
    }
    
    _create_readme(output_path)
    
    print("\nDataset preprocessing completed successfully!")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples") 
    print(f"Test: {len(test_df)} samples")
    print(f"Classes: {sorted(df['label'].unique())}")


def main():
    """Main function to run preprocessing with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess VIVAE dataset for vocal classification"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to VIVAE dataset directory containing VIVAE folder"
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Training set proportion (default: 0.7)"
    )
    parser.add_argument(
        "--test-size", 
        type=float,
        default=0.15,
        help="Test set proportion (default: 0.15)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per class (default: 10)"
    )
    
    args = parser.parse_args()
    
    preprocess_vivae(
        data_path=args.data_path,
        train_size=args.train_size,
        test_size=args.test_size,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()