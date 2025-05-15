from it_multimodal_er.preprocessing import PreprocessorFactory, PreprocessorConfig
from datasets import Dataset

def load_dataset(preprocessing_config: PreprocessorConfig) -> Dataset:
    """
    Load and preprocess the dataset.

    args:
        - preprocessing_config: PreprocessorConfig, configuration for the preprocessor.

    returns:
        - dataset: Dataset, preprocessed dataset.
    """

    # Initialize preprocessor
    preprocessor = PreprocessorFactory.get_preprocessor(
        preprocessor_config=preprocessing_config
    )

    # Preprocess dataset
    dataset = preprocessor.preprocess()

    # Shuffle dataset
    dataset = dataset.shuffle()

    return dataset