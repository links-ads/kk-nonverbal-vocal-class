_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=5,
    class_weights=None,
    label2id={
        "hungry":0,
        "discomfort":1,
        "belly_pain":2,
        "tired":3,
    },
    id2label={
        0:"hungry",
        1:"discomfort",
        2:"belly_pain",
        3:"tired",
    }
)


preprocessing_config=dict(
    datasets_path="donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/",
    audio_dataset_path="samples/",
    dataset_name="donateacry",
    label2id={
        "hungry": 0,
        "discomfort": 1,
        "belly_pain": 2,
        "tired": 3
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="wav2vec2-donateacry-adapter",
    save_path="./outputs/donateacry",
)