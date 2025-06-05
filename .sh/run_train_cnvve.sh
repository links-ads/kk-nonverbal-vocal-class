#!/bin/bash

sbatch jobs/cnvve/cnvve_wavlm_base_plus_lora.job
sbatch jobs/cnvve/cnvve_whisper_base_frozen.job
sbatch jobs/cnvve/cnvve_wavlm_base_plus_frozen.job
sbatch jobs/cnvve/cnvve_whisper_base_lora.job
sbatch jobs/cnvve/cnvve_wav2vec2_base_frozen.job
sbatch jobs/cnvve/cnvve_hubert_base_frozen.job
sbatch jobs/cnvve/cnvve_wav2vec2_base_lora.job
sbatch jobs/cnvve/cnvve_hubert_base_lora.job
