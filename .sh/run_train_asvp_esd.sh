#!/bin/bash

sbatch jobs/asvp_esd/asvp_esd_whisper_base_lora.job
sbatch jobs/asvp_esd/asvp_esd_wavlm_base_plus_frozen.job
sbatch jobs/asvp_esd/asvp_esd_wav2vec2_base_frozen.job
sbatch jobs/asvp_esd/asvp_esd_whisper_base_frozen.job
sbatch jobs/asvp_esd/asvp_esd_hubert_base_lora.job
sbatch jobs/asvp_esd/asvp_esd_wav2vec2_base_lora.job
sbatch jobs/asvp_esd/asvp_esd_hubert_base_frozen.job
sbatch jobs/asvp_esd/asvp_esd_wavlm_base_plus_lora.job
