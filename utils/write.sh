python utils/write_jobs.py experiments_configs/cnvve jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/asvp_esd jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/asvp_esd_babies jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/donateacry jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/nonverbal_vocalization_dataset jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/recanvo jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/vivae jobs/__base__/train.job jobs/

python utils/write_sh.py jobs/ .sh/run_whisper_partial.sh