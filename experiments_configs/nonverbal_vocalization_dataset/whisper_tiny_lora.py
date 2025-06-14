_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="whisper",
    audio_model_name="openai/whisper-tiny",
    finetune_method="lora",
    num_labels=16,
    class_weights=None,
    label2id={
        "throat-clearing": 0,
        "lip-popping": 1,
        "sneezing": 2,
        "sighing": 3,
        "screaming": 4,
        "coughing": 5,
        "nose-blowing": 6,
        "yawning": 7,
        "tongue-clicking": 8,
        "teeth-chattering": 9,
        "lip-smacking": 10,
        "moaning": 11,
        "panting": 12,
        "teeth-grinding": 13,
        "laughing": 14,
        "crying": 15
    },
    id2label={
        0: "throat-clearing",
        1: "lip-popping",
        2: "sneezing",
        3: "sighing",
        4: "screaming",
        5: "coughing",
        6: "nose-blowing",
        7: "yawning",
        8: "tongue-clicking",
        9: "teeth-chattering",
        10: "lip-smacking",
        11: "moaning",
        12: "panting",
        13: "teeth-grinding",
        14: "laughing",
        15: "crying"
    }
)

preprocessing_config=dict(
    datasets_path="nonverbal_vocalization_dataset/",
    audio_dataset_path="samples/",
    dataset_name="nonverbal_vocalization_dataset",
    label2id={
        "throat-clearing": 0,
        "lip-popping": 1,
        "sneezing": 2,
        "sighing": 3,
        "screaming": 4,
        "coughing": 5,
        "nose-blowing": 6,
        "yawning": 7,
        "tongue-clicking": 8,
        "teeth-chattering": 9,
        "lip-smacking": 10,
        "moaning": 11,
        "panting": 12,
        "teeth-grinding": 13,
        "laughing": 14,
        "crying": 15
    },
    audio_model_name="openai/whisper-tiny",
)

training_config=dict(
    output_model_name="nonverbal_vocalization_dataset_whisper_tiny_lora",
    experiment_dir="nonverbal_vocalization_dataset/whisper_tiny_lora",
)