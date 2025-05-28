import os
from typing import Optional
from dataclasses import dataclass

from .model_config import ModelConfig
from .preprocessing_config import PreprocessingConfig
from .training_config import TrainingConfig

from mmengine import Config as MMEngineConfig

@dataclass
class Config:    
    model_config: Optional[ModelConfig] = None
    preprocessing_config: Optional[PreprocessingConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            cfg = MMEngineConfig.fromfile(config_path)
        except:
            raise FileNotFoundError(f"Invalid configuration file: {config_path}")
        
        # Initialize configs
        model_config = None
        preprocessing_config = None
        training_config = None
        
        # Load model config
        if hasattr(cfg, 'model_config'):
            model_config = ModelConfig.from_dict(cfg.model_config)
        
        # Load preprocessing config
        if hasattr(cfg, 'preprocessing_config'):
            preprocessing_config = PreprocessingConfig(config=cfg.preprocessing_config)
        
        # Load training config
        if hasattr(cfg, 'training_config'):
            training_config = TrainingConfig(config=cfg.training_config)
        
        return cls(
            model_config=model_config,
            preprocessing_config=preprocessing_config,
            training_config=training_config
        )
    
    def __repr__(self) -> str:
        """String representation of the Config instance."""
        return (f"Config(\n"
                f"  model_config={self.model_config is not None},\n"
                f"  preprocessing_config={self.preprocessing_config is not None},\n"
                f"  training_config={self.training_config is not None}\n"
                f")")