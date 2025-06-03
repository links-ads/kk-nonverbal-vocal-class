_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=7,
    class_weights=None,
    label2id={
        "ahem": 0,
        "samples": 1,
        "confirm": 2,
        "continuous": 3,
        "decline": 4,
        "hush": 5,
        "psst": 6
    },
    id2label={
        0:"ahem",
        1:"samples",
        2:"confirm",
        3:"continuous",
        4:"decline",
        5:"hush",
        6:"psst"
    }
)


preprocessing_config=dict(
    datasets_path="cnvve/",
    audio_dataset_path="samples/",
    dataset_name="cnvve",
    label2id={
        "ahem": 0,
        "samples": 1,
        "confirm": 2,
        "continuous": 3,
        "decline": 4,
        "hush": 5,
        "psst": 6
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="wav2vec2-cnvve-adapter",
    save_path="./outputs/cnvve",
)