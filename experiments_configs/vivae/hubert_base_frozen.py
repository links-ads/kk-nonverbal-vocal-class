_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="hubert",
    audio_model_name="facebook/hubert-base-ls960",
    finetune_method="frozen",
    num_labels=6,
    class_weights=None,
    label2id={
        "achievement": 0,
        "anger": 1,
        "fear": 2,
        "pain": 3,
        "pleasure": 4,
        "surprise": 5
    },
    id2label={
        0: "achievement",
        1: "anger",
        2: "fear",
        3: "pain",
        4: "pleasure",
        5: "surprise"
    }
)

preprocessing_config=dict(
    datasets_path="vivae/",
    audio_dataset_path="samples/",
    dataset_name="vivae",
    label2id={
        "ahem": 0,
        "confirm": 1,
        "continuous": 2,
        "decline": 3,
        "hush": 4,
        "psst": 5
    },
    audio_model_name="facebook/hubert-base-ls960",
)

training_config=dict(
    output_model_name="vivae_hubert_base_frozen",
    experiment_dir="vivae/hubert_base_frozen",
)