_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="whisper",
    audio_model_name="openai/whisper-base",
    finetune_method="lora",
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
    datasets_path="asvp_esd/",
    audio_dataset_path="samples/",
    dataset_name="asvp_esd",
    label2id={
        "ahem": 0,
        "confirm": 1,
        "continuous": 2,
        "decline": 3,
        "hush": 4,
        "psst": 5
    },
    audio_model_name="openai/whisper-base",
)

training_config=dict(
    output_model_name="asvp_esd_whisper_base_lora",
    experiment_dir="asvp_esd/whisper_base_lora",
)