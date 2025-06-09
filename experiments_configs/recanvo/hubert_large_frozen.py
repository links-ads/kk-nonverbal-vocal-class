_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

LABELS = [
    'selftalk', 'frustrated', 'delighted', 'dysregulated', 'social', 'request'
]

model_config = dict(
    model_type="hubert",
    audio_model_name="facebook/hubert-large-ls960-ft",
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
    audio_model_name="facebook/hubert-large-ls960-ft",
)

training_config=dict(
    output_model_name="recanvo_hubert_large_frozen",
    experiment_dir="recanvo/hubert_large_frozen",
)