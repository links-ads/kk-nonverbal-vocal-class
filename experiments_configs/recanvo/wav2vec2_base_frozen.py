_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=6,
    class_weights=None,
    label2id={
        "ahem": 0,
        "confirm": 1,
        "continuous": 2,
        "decline": 3,
        "hush": 4,
        "psst": 5
    },
    id2label={
        0:"ahem",
        1:"confirm",
        2:"continuous",
        3:"decline",
        4:"hush",
        5:"psst"
    }
)

preprocessing_config=dict(
    datasets_path="recanvo/",
    audio_dataset_path="samples/",
    dataset_name="recanvo",
    label2id={
        "ahem": 0,
        "confirm": 1,
        "continuous": 2,
        "decline": 3,
        "hush": 4,
        "psst": 5
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="recanvo_wav2vec2_base_frozen",
    experiment_dir="recanvo/wav2vec2_base_frozen",
)