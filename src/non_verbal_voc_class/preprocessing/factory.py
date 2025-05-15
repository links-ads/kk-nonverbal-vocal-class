
from non_verbal_voc_class.preprocessing import AudioPreprocessor
from non_verbal_voc_class.preprocessing import PreprocessorConfig

class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(preprocessor_config: PreprocessorConfig):
        """
        Factory function to get the appropriate preprocessor based on the type.

        Parameters:
            preprocessor_config (PreprocessorConfig): Configuration object for the preprocessor.
        Returns:
            Preprocessor: An instance of the appropriate preprocessor class.
        Raises:
            ValueError: If the preprocessor_type is unknown.              
        """

        if preprocessor_config.preprocessor_type == "audio":
            return AudioPreprocessor(preprocessor_config)
        else:
            error_message = f"Unknown preprocessor type: {preprocessor_config.preprocessor_type}"
            raise ValueError(error_message)