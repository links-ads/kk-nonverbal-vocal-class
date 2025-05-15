import torch
from transformers import (
    AutoConfig,
    ASTForAudioClassification,
)
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTMLPHead
from .model import CustomModelForAudioClassification

AVAILABLE_MODELS = [
    "facebook/wav2vec2-large",
    "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "microsoft/wavlm-large",
    "openai/whisper-small",
    "openai/whisper-medium",
    ]

def make_model(hparams):
    """ Returns a model instance based on the provided hyperparameters. """

    # hparams = vars(hparams)
    hparams = hparams.__dict__

    if hparams['HF_MODEL_PATH'] == "MIT/ast-finetuned-audioset-10-10-0.4593":
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model.config.num_labels = hparams['NUM_LABELS']
        model.config.label_weights = hparams['LABEL_WEIGHTS']
        # config.output_hidden_states = hparams['OUTPUT_HIDDEN_STATES']
        model.classifier = ASTMLPHead(model.config)
    else:
        # assert hparams['HF_MODEL_PATH'] in AVAILABLE_MODELS, f"Model {hparams['HF_MODEL_PATH']} not available. Choose from {AVAILABLE_MODELS}"
        # Add hparams to the config object
        config = AutoConfig.from_pretrained(hparams['HF_MODEL_PATH'])
        config.max_duration = hparams['MAX_DURATION']
        config.sampling_rate = hparams['SAMPLING_RATE']
        config.output_hidden_states = hparams['OUTPUT_HIDDEN_STATES']
        config.classifier_name = hparams['CLASSIFIER_NAME']
        config.classifier_proj_size = hparams['CLASSIFIER_PROJ_SIZE']
        config.num_labels = hparams['NUM_LABELS']
        config.label_weights = hparams['LABEL_WEIGHTS']
        config.lossname = hparams['LOSS']
        # config.label_weights = hparams['LABEL_WEIGHTS'] # TODO add label weights to config

        # Instantiate the model
        model = CustomModelForAudioClassification(config)
    # model.to(
    #     # hparams['GPU_ID']
    # )
    
    return model