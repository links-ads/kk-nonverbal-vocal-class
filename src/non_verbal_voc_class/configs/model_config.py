import json

from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(self, path_to_config: str = None, **kwargs):
        if path_to_config is not None:
            with open(path_to_config, 'r') as f:
                config_data = json.load(f)
            
            for key, value in config_data.items():
                setattr(self, key, value)
        super().__init__(**kwargs)