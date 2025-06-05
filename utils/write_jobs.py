"""
Run with: 
python utils/write_jobs.py <config_directory> <job_template> <output_directory>

python utils/write_jobs.py experiments_configs/cnvve jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/asvp_esd jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/asvp_esd_babies jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/donateacry jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/nonverbal_vocalization_dataset jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/recanvo jobs/__base__/train.job jobs/
python utils/write_jobs.py experiments_configs/vivae jobs/__base__/train.job jobs/
"""

#!/bin/bash python3

import os
import sys

def update_job_file(
        template_path,
        config_path,
        job_name,
        log_path,
    ):
    with open(template_path, 'r') as file:
        job_script = file.read()

    job_script = job_script.replace('--job-name=job_name', f'--job-name={job_name}')
    job_script = job_script.replace('--output=path/to/log.out', f'--output={log_path}')
    job_script = job_script.replace('python -BO train.py path/to/config.py', f'python -BO train.py {config_path}')

    return job_script

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_training.py <config_directory> <job_template> <output_directory>")
        sys.exit(1)

    config_directory = sys.argv[1]
    job_template = sys.argv[2]
    output_directory = os.path.join(sys.argv[3], os.path.basename(config_directory))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for config_file in os.listdir(config_directory):
        if config_file.endswith('.py'):
            config_path = os.path.join(config_directory, config_file)
            job_name = os.path.basename(config_directory)+'_'+os.path.splitext(config_file)[0]
            log_path = os.path.join('logs', os.path.basename(config_directory), f'{job_name}.out')

            job_script = update_job_file(job_template, config_path, job_name, log_path)
            job_file_path = os.path.join(output_directory, f'{job_name}.job')

            with open(job_file_path, 'w') as job_file:
                job_file.write(job_script)

if __name__ == "__main__":
    main()
