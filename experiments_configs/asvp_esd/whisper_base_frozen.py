_base_ = [
    '../_base_/base_model.py',
    '../_base_/base_preprocessing.py',
    '../_base_/base_training.py',
]

model_config = dict(
    model_type="whisper",
    audio_model_name="openai/whisper-base",
    finetune_method="frozen",
    num_labels=14,
    class_weights=None,
    label2id={
        "boredom_sigh": 0,
        "neutral_calm": 1,
        "happy_laugh_gaggle": 2,
        "sad_cry": 3,
        "angry_grunt_frustration": 4,
        "fearful_scream_panic": 5,
        "disgust_dislike_contempt": 6,
        "surprised_gasp_amazed": 7,
        "excited": 8,
        "pleasure": 9,
        "pain_groan": 10,
        "disappointment_disapproval": 11,
        "breath": 12,
        "disgust_dislike_contempt": 13
    },
    id2label={
        0: "boredom_sigh",
        1: "neutral_calm",
        2: "happy_laugh_gaggle",
        3: "sad_cry",
        4: "angry_grunt_frustration",
        5: "fearful_scream_panic",
        6: "disgust_dislike_contempt",
        7: "surprised_gasp_amazed",
        8: "excited",
        9: "pleasure",
        10: "pain_groan",
        11: "disappointment_disapproval",
        12: "breath",
        13: "disgust_dislike_contempt"
    }
)

preprocessing_config=dict(
    datasets_path="asvp_esd/",
    audio_dataset_path="samples/",
    dataset_name="asvp_esd",
    label2id={
        "boredom_sigh": 0,
        "neutral_calm": 1,
        "happy_laugh_gaggle": 2,
        "sad_cry": 3,
        "angry_grunt_frustration": 4,
        "fearful_scream_panic": 5,
        "disgust_dislike_contempt": 6,
        "surprised_gasp_amazed": 7,
        "excited": 8,
        "pleasure": 9,
        "pain_groan": 10,
        "disappointment_disapproval": 11,
        "breath": 12,
        "disgust_dislike_contempt": 13
    },
    audio_model_name="openai/whisper-base",
)

training_config=dict(
    output_model_name="asvp_esd_whisper_base_frozen",
    experiment_dir="asvp_esd/whisper_base_frozen",
)