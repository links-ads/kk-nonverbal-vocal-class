_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

LABELS = [
    'selftalk', 'frustrated', 'delighted', 'dysregulated', 'social', 'request'
]

model_config = dict(
    model_type="whisper",
    audio_model_name="openai/whisper-small",
    finetune_method="frozen",
    num_labels=len(LABELS),
    class_weights=None,
    label2id={label: idx for idx, label in enumerate(LABELS)},
    id2label={idx: label for idx, label in enumerate(LABELS)}
)

preprocessing_config=dict(
    datasets_path="recanvo/",
    audio_dataset_path="samples/",
    dataset_name="recanvo",
    label2id={label: idx for idx, label in enumerate(LABELS)},
    audio_model_name="openai/whisper-small",
)

training_config=dict(
    output_model_name="recanvo_whisper_small_frozen",
    experiment_dir="recanvo/whisper_small_frozen",
)