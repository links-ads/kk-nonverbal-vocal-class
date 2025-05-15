from non_verbal_voc_class.training import get_loss_function
from non_verbal_voc_class.configs import ModelConfig
from transformers.modeling_outputs import ModelOutput
from torch import Tensor
from typing import Tuple
from transformers import (
    PreTrainedModel,
    AutoConfig
)
from .factory import (
    AudioModelFactory,
    ClassifierFactory
)

class AudioModelForClassification(PreTrainedModel):
    config_class = ModelConfig
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.audio_model = AudioModelFactory.create_audio_model(
            model_type=config.audio_model_type,
            model_config=config
        )
        self.audio_model_config = self.audio_model.model_config
        self.audio_model_config.num_labels = config.num_classes
        # self.audio_model_config.id2label = config.id2label
        # self.audio_model_config.label2id = config.label2id
        # self.audio_model.freeze_feature_encoder()
        # self.audio_model.freeze_parameters()

        self.classifier = ClassifierFactory.create_classifier(
            classifier_type=config.classifier_type,
            model_config=self.audio_model_config
        )

        tensor_label_weights = Tensor(config.label_weights)
        self.loss_fnct = get_loss_function(tensor_label_weights)

    def forward(
            self, 
            input_features: Tensor, 
            attention_mask: Tensor = None, 
            length: Tensor = None,
            encoder_outputs: Tuple[Tuple[Tensor]] = None,
            labels=None
        ) -> ModelOutput:
        """
            Args:
            ------
                - input_features: the input features for the audio model
                - attention_mask: the attention mask for the audio model
                - length: the length of the input features
                - encoder_outputs: the hidden states of the audio model
                - labels: the labels for the classification task

            Returns:
            --------
                - ModelOutput: the output of the model
        """
        if encoder_outputs is None:
            encoder_output = self.audio_model(
                input_features,
                length=length,
            )

        logits = self.classifier(**encoder_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fnct(
                input=logits.view(-1, self.audio_model_config.num_labels), 
                target=labels.view(-1)
            )


        return ModelOutput(
            logits=logits,
            loss=loss
        )