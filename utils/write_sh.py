"""
Run with: 
python utils/write_sh.py <jobs_folder> <output_script>

python utils/write_sh.py jobs/cnvve .sh/run_train_cnvve.sh
python utils/write_sh.py jobs/asvp_esd .sh/run_train_asvp_esd.sh
python utils/write_sh.py jobs/asvp_esd_babies .sh/run_train_asvp_esd_babies.sh
python utils/write_sh.py jobs/donateacry .sh/run_train_donateacry.sh
python utils/write_sh.py jobs/nonverbal_vocalization_dataset .sh/run_train_nonverbal_vocalization_dataset.sh
python utils/write_sh.py jobs/recanvo .sh/run_train_recanvo.sh
python utils/write_sh.py jobs/vivae .sh/run_train_vivae.sh
"""


import os
import sys

def generate_run_training_script(jobs_folder, output_script):
    with open(output_script, 'w') as script_file:
        script_file.write("#!/bin/bash\n\n")
        
        # Walk through the directory tree to find all .job files
        for root, _, files in os.walk(jobs_folder):
            for file in files:
                # Skip files that are not .job or are named train.job
                if file.endswith('.job') and file != 'train.job':
                    job_path = os.path.join(root, file)
                    script_file.write(f"sbatch {job_path}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python write_sh.py <jobs_folder> <output_script>")
        sys.exit(1)

    # Create the .sh folder if it doesn't exist
    os.makedirs('.sh', exist_ok=True)

    jobs_folder = sys.argv[1]
    output_script = sys.argv[2]

    generate_run_training_script(jobs_folder, output_script)
    os.chmod(output_script, 0o755)  # Make the script executable

if __name__ == "__main__":
    main()