from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from ..configs.model_config import ModelConfig

class BaseClassifier(PreTrainedModel):
    
    def __init__(self, config: ModelConfig):
        super(BaseClassifier, self).__init__(config)
        
        assert config.finetune_method in ["adapter", "adapter_l", "embedding_prompt", "lora", "combined", "finetune", "frozen"], "finetune method not available"
        assert hasattr(config, 'model_type'), "Model type must be specified in the config"
        
        self.config = config
        self.finetune_method = self.config.finetune_method
        
        self.class_weights = config.class_weights if hasattr(config, 'class_weights') else None
        self.loss_fct = CrossEntropyLoss(weight=self.class_weights) 

    def _add_adapter_and_freeze(self) -> None:
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def forward(self, *args, **kwargs) -> SequenceClassifierOutput:
        raise NotImplementedError("This method must be implemented by subclasses")