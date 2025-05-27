import json
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .model_config import ModelConfig
from .preprocessing_config import PreprocessingConfig
from .training_config import TrainingConfig


@dataclass
class Config:
    """
    Main configuration class that contains model, preprocessing, and training configurations.
    
    This class can read from a JSON file with the following structure:
    {
        "model_config": {
            "model_type": "hubert",
            "audio_model_name": "facebook/hubert-base-ls960",
            ...
        },
        "preprocessing_config": {
            "target_sampling_rate": 16000,
            "max_duration": 5.0,
            ...
        },
        "training_config": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            ...
        }
    }
    """
    
    model_config: Optional[ModelConfig] = None
    preprocessing_config: Optional[PreprocessingConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    @classmethod
    def from_json(cls, json_file_path: str) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            json_file_path: Path to the JSON configuration file
            
        Returns:
            Config instance with loaded configurations
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON file is invalid or missing required sections
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Configuration file not found: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")
        
        # Initialize configs
        model_config = None
        preprocessing_config = None
        training_config = None
        
        # Load model config
        if 'model_config' in data:
            model_config = cls._create_model_config(data['model_config'])
        
        # Load preprocessing config
        if 'preprocessing_config' in data:
            preprocessing_config = PreprocessingConfig(config=data['preprocessing_config'])
        
        # Load training config
        if 'training_config' in data:
            training_config = TrainingConfig(config=data['training_config'])
        
        return cls(
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            training_config=training_config
        )
    
    @staticmethod
    def _create_model_config(model_config_data: Dict[str, Any]) -> ModelConfig:
        """Create a ModelConfig instance from dictionary data."""
        config = ModelConfig()
        for key, value in model_config_data.items():
            setattr(config, key, value)
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration data
            
        Returns:
            Config instance
        """
        model_config = None
        preprocessing_config = None
        training_config = None
        
        if 'model_config' in config_dict:
            model_config = cls._create_model_config(config_dict['model_config'])
        
        if 'preprocessing_config' in config_dict:
            preprocessing_config = PreprocessingConfig(config=config_dict['preprocessing_config'])
        
        if 'training_config' in config_dict:
            training_config = TrainingConfig(config=config_dict['training_config'])
        
        return cls(
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            training_config=training_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Config instance to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        
        if self.model_config is not None:
            # Convert ModelConfig to dict (it inherits from PretrainedConfig)
            result['model_config'] = self.model_config.to_dict()
        
        if self.preprocessing_config is not None:
            result['preprocessing_config'] = self.preprocessing_config.config
        
        if self.training_config is not None:
            result['training_config'] = self.training_config.config
        
        return result
    
    def to_json(self, json_file_path: str, indent: int = 2) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            json_file_path: Path where to save the JSON file
            indent: JSON indentation level
        """
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(config_dict, file, indent=indent, ensure_ascii=False)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Basic validation - at least one config should be present
        if (self.model_config is None and 
            self.preprocessing_config is None and 
            self.training_config is None):
            return False
        
        # Add more specific validation logic here if needed
        return True
    
    def get_model_config(self) -> Optional[ModelConfig]:
        """Get the model configuration."""
        return self.model_config
    
    def get_preprocessing_config(self) -> Optional[PreprocessingConfig]:
        """Get the preprocessing configuration."""
        return self.preprocessing_config
    
    def get_training_config(self) -> Optional[TrainingConfig]:
        """Get the training configuration."""
        return self.training_config
    
    def update_model_config(self, **kwargs) -> None:
        """Update model configuration with new values."""
        if self.model_config is None:
            self.model_config = ModelConfig()
        
        for key, value in kwargs.items():
            setattr(self.model_config, key, value)
    
    def update_preprocessing_config(self, **kwargs) -> None:
        """Update preprocessing configuration with new values."""
        if self.preprocessing_config is None:
            self.preprocessing_config = PreprocessingConfig()
        
        self.preprocessing_config.config.update(kwargs)
    
    def update_training_config(self, **kwargs) -> None:
        """Update training configuration with new values."""
        if self.training_config is None:
            self.training_config = TrainingConfig()
        
        self.training_config.config.update(kwargs)
    
    def __repr__(self) -> str:
        """String representation of the Config instance."""
        return (f"Config(\n"
                f"  model_config={self.model_config is not None},\n"
                f"  preprocessing_config={self.preprocessing_config is not None},\n"
                f"  training_config={self.training_config is not None}\n"
                f")")