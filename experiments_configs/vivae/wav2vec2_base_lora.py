_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="lora",
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
        "achievement": 0,
        "anger": 1,
        "fear": 2,
        "pain": 3,
        "pleasure": 4,
        "surprise": 5
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="vivae_wav2vec2_base_lora",
    experiment_dir="vivae/wav2vec2_base_lora",
)