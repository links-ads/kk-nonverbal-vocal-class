from .classifiers import (
    LinearClassifier,
    MultiLevelClassifier,
    NonLinearClassifier,
    AudioModelClassifier,
)
from transformers import PretrainedConfig


# TODO: Implement the BaseClassifier class
class ClassifierFactory:
    @staticmethod
    def create_classifier(classifier_type: str, model_config: PretrainedConfig) -> object:
        """
            Factory function to get the appropriate downstream classifier based on the model type.
        
            Parameters:
            ----------
                classifier_type (str): The type of the downstream classifier. Must be one of 'linear_classifier', 'non_linear_classifier', or 'multilevel_classifier'.
                model_config (PretrainedConfig): The configuration for the model.

            Returns:
            --------
                object: An instance of the downstream classifier based on the model type.

            Raises:
            -------
                ValueError: If the classifier_type is unknown.
        """
        if classifier_type == 'linear_classifier':
            return LinearClassifier(model_config)
        elif classifier_type == 'non_linear_classifier':
            return NonLinearClassifier(model_config)
        elif classifier_type == 'multilevel_classifier':
            return MultiLevelClassifier(model_config)
        elif classifier_type == 'audio_model_classifier':
            return AudioModelClassifier(model_config)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")