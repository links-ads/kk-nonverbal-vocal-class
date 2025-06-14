#!/bin/bash

sbatch jobs/asvp_esd/asvp_esd_hubert_large_frozen.job
sbatch jobs/asvp_esd/asvp_esd_wav2vec2_large_frozen.job
sbatch jobs/asvp_esd/asvp_esd_wavlm_large_frozen.job
sbatch jobs/asvp_esd/asvp_esd_whisper_medium_frozen.job
sbatch jobs/asvp_esd/asvp_esd_whisper_large_frozen.job

sbatch jobs/vivae/vivae_wav2vec2_large_frozen.job
sbatch jobs/vivae/vivae_wavlm_large_frozen.job
sbatch jobs/vivae/vivae_hubert_large_frozen.job
sbatch jobs/vivae/vivae_whisper_medium_frozen.job
sbatch jobs/vivae/vivae_whisper_large_frozen.job

sbatch jobs/recanvo/recanvo_hubert_large_frozen.job
sbatch jobs/recanvo/recanvo_wav2vec2_large_frozen.job
sbatch jobs/recanvo/recanvo_wavlm_large_frozen.job
sbatch jobs/recanvo/recanvo_whisper_large_frozen.job
sbatch jobs/recanvo/recanvo_whisper_medium_frozen.job
