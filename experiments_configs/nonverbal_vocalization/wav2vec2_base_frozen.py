_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=16,
    class_weights=None,
    label2id={
        "teeth-grinding": 0,
        "screaming": 1,
        "sighing": 2,
        "lip-smacking": 3,
        "teeth-chattering": 4,
        "sneezing": 5,
        "panting": 6,
        "tongue-clicking": 7,
        "laughing": 8,
        "nose-blowing": 9,
        "moaning": 10,
        "coughing": 11,
        "crying": 12,
        "lip-popping": 13,
        "throat-clearing": 14,
        "yawning": 15
        },
    id2label={
        0:"teeth-grinding",
        1:"screaming",
        2:"sighing",
        3:"lip-smacking",
        4:"teeth-chattering",
        5:"sneezing",
        6:"panting",
        7:"tongue-clicking",
        8:"laughing",
        9:"nose-blowing",
        10:"moaning",
        11:"coughing",
        12:"crying",
        13:"lip-popping",
        14:"throat-clearing",
        15:"yawning"
    }
)


preprocessing_config=dict(
    datasets_path="nonverbal_vocalization_dataset/NonverbalVocalization/",
    audio_dataset_path="samples/",
    dataset_name="nonverbal-vocalization",
    label2id={
        "teeth-grinding": 0,
        "screaming": 1,
        "sighing": 2,
        "lip-smacking": 3,
        "teeth-chattering": 4,
        "sneezing": 5,
        "panting": 6,
        "tongue-clicking": 7,
        "laughing": 8,
        "nose-blowing": 9,
        "moaning": 10,
        "coughing": 11,
        "crying": 12,
        "lip-popping": 13,
        "throat-clearing": 14,
        "yawning": 15
        },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="wav2vec2-nonverbal-vocalization-adapter",
    save_path="./outputs/nonverbal_vocalization",
)