#!/bin/bash

sbatch jobs/recanvo/recanvo_wav2vec2_base_frozen.job
sbatch jobs/recanvo/recanvo_wav2vec2_base_lora.job
sbatch jobs/recanvo/recanvo_wavlm_base_plus_frozen.job
sbatch jobs/recanvo/recanvo_whisper_base_lora.job
sbatch jobs/recanvo/recanvo_whisper_base_frozen.job
sbatch jobs/recanvo/recanvo_wavlm_base_plus_lora.job
sbatch jobs/recanvo/recanvo_hubert_base_lora.job
sbatch jobs/recanvo/recanvo_hubert_base_frozen.job
