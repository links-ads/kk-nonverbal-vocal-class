_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=17,
    class_weights=None,
    label2id={
        "selftalk": 0,
        "social": 1,
        "frustrated": 2,
        "delighted": 3,
        "more": 4,
        "affectionate": 5,
        "request": 6,
        "protest": 7,
        "yes": 8,
        "dysregulated": 9,
        "bathroom": 10,
        "happy": 11,
        "laughter": 12,
        "dysregulation-sick": 13,
        "dysregulation-bathroom": 14,
        "help": 15,
        "no": 16
    },
    id2label={
        0:"selftalk",
        1:"social",
        2:"frustrated",
        3:"delighted",
        4:"more",
        5:"affectionate",
        6:"request",
        7:"protest",
        8:"yes",
        9:"dysregulated",
        10:"bathroom",
        11:"happy",
        12:"laughter",
        13:"dysregulation-sick",
        14:"dysregulation-bathroom",
        15:"help",
        16:"no"
    }
)


preprocessing_config=dict(
    datasets_path="ReCANVo/",
    audio_dataset_path="samples",
    dataset_name="recanvo",
    label2id={
        "selftalk": 0,
        "social": 1,
        "frustrated": 2,
        "delighted": 3,
        "more": 4,
        "affectionate": 5,
        "request": 6,
        "protest": 7,
        "yes": 8,
        "dysregulated": 9,
        "bathroom": 10,
        "happy": 11,
        "laughter": 12,
        "dysregulation-sick": 13,
        "dysregulation-bathroom": 14,
        "help": 15,
        "no": 16
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="wav2vec2_base-recanvo-frozen",
    save_path="./outputs/recanvo",
)