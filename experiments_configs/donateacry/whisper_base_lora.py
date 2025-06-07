_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="whisper",
    audio_model_name="openai/whisper-base",
    finetune_method="lora",
    num_labels=5,
    class_weights=None,
    label2id={
        "belly_pain": 0,
        "burping": 1,
        "discomfort": 2,
        "hungry": 3,
        "tired": 4,
    },
    id2label={
        0:"belly_pain",
        1:"burping",
        2:"discomfort",
        3:"hungry",
        4:"tired",
    }
)

preprocessing_config=dict(
    datasets_path="donateacry_corpus/",
    audio_dataset_path="samples/",
    dataset_name="donateacry",
    label2id={
        "belly_pain": 0,
        "burping": 1,
        "discomfort": 2,
        "hungry": 3,
        "tired": 4,
    },
    audio_model_name="openai/whisper-base",
)

training_config=dict(
    output_model_name="donateacry_whisper_base_lora",
    experiment_dir="donateacry/whisper_base_lora",
)