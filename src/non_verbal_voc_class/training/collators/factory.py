from .audio_collator import AudioCollator
from .base_collator import BaseCollator

class CollatorFactory:
    @staticmethod
    def create_collator(collator_type: str) -> BaseCollator:
        """
        Factory function to get the appropriate collator based on the collator type.
    
        Parameters:
        ----------
        collator_type (str): The type of the collator. Must be one of "audio" or "multimodal".
        collator_config (PretrainedConfig): The configuration for the collator.

        Returns:
        --------
        BaseCollator: An instance of the collator based on the collator type.

        Raises:
        -------
        ValueError: If the collator_type is unknown.
        """
        if collator_type == "audio" or collator_type == "old_audio":
            return AudioCollator()
        else:
            raise ValueError(f"Unknown collator type: {collator_type}")