from non_verbal_voc_class.preprocessing import AudioPreprocessor, PreprocessorConfig
from datasets import Dataset

def load_dataset(preprocessing_config: PreprocessorConfig) -> Dataset:
    """
    Load and preprocess the dataset.

    args:
        - preprocessing_config: PreprocessorConfig, configuration for the preprocessor.

    returns:
        - dataset: Dataset, preprocessed dataset.
    """
    preprocessor = AudioPreprocessor(preprocessing_config)

    dataset = preprocessor.preprocess()

    dataset = dataset.shuffle()

    return dataset