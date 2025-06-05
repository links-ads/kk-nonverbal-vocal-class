#!/bin/bash

sbatch jobs/vivae/vivae_whisper_base_lora.job
sbatch jobs/vivae/vivae_wavlm_base_plus_lora.job
sbatch jobs/vivae/vivae_wavlm_base_plus_frozen.job
sbatch jobs/vivae/vivae_whisper_base_frozen.job
sbatch jobs/vivae/vivae_hubert_base_frozen.job
sbatch jobs/vivae/vivae_wav2vec2_base_frozen.job
sbatch jobs/vivae/vivae_wav2vec2_base_lora.job
sbatch jobs/vivae/vivae_hubert_base_lora.job
