_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="wav2vec2",
    audio_model_name="facebook/wav2vec2-base",
    finetune_method="frozen",
    num_labels=13,
    class_weights=None,
    label2id={
        "boredom": 0,
        "pain": 1,
        "happy": 2,
        "sad": 3,
        "fearful": 4,
        "surprised": 5,
        "excited": 6,
        "disgust": 7,
        "pleasure": 8,
        "breath": 9,
        "neutral": 10,
        "angry": 11,
        "disappointment": 12
    },
    id2label={
        0:"boredom",
        1:"pain",
        2:"happy",
        3:"sad",
        4:"fearful",
        5:"surprised",
        6:"excited",
        7:"disgust",
        8:"pleasure",
        9:"breath",
        10:"neutral",
        11:"angry",
        12:"disappointment",
    }
)


preprocessing_config=dict(
    datasets_path="asvp-esd/",
    audio_dataset_path="samples/",
    dataset_name="asvp-esd",
    label2id={
        "boredom": 0,
        "pain": 1,
        "happy": 2,
        "sad": 3,
        "fearful": 4,
        "surprised": 5,
        "excited": 6,
        "disgust": 7,
        "pleasure": 8,
        "breath": 9,
        "neutral": 10,
        "angry": 11,
        "disappointment": 12
    },
    audio_model_name="facebook/wav2vec2-base",
)

training_config=dict(
    output_model_name="wav2vec2-asvp-esd-adapter",
    save_path="./outputs/asvp-esd",
)