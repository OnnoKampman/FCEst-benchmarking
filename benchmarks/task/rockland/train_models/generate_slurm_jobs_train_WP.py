import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.rockland import get_rockland_subjects
from helpers.slurm import save_slurm_job_to_file, send_slurm_job_to_queue


if __name__ == "__main__":

    hostname = socket.gethostname()
    assert hostname == "hivemind"  # all parallel jobs should be run on the Hivemind

    data_set_name = 'rockland'
    pp_pipeline = 'custom_fsl_pipeline'

    model_name = sys.argv[1]  # 'VWP_joint' or 'SVWP_joint'
    data_split = sys.argv[2]       # 'all' or 'LEOO'
    repetition_time = sys.argv[3]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        subset=repetition_time,
        hostname=hostname
    )
    roi_list_name = cfg['roi-list-name']
    experiments_basedir = cfg['experiments-basedir']
    max_number_of_cpus = cfg['max-n-cpus']
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    n_subjects = len(all_subjects_list)

    slurm_job_str = ("#!/bin/bash\n"
                     f"#SBATCH --job-name={data_set_name:s}_{repetition_time:s}_{roi_list_name:s}_{data_split:s}_{model_name:s}\n"
                     f"#SBATCH --output={experiments_basedir:s}/{pp_pipeline:s}/slurm_logs/{data_split:s}/{model_name:s}/%A_%a.out\n"
                     f"#SBATCH --error={experiments_basedir:s}/{pp_pipeline:s}/slurm_logs/{data_split:s}/{model_name:s}/%A_%a.err\n"
                     "#SBATCH --nodes=1\n"
                     "#SBATCH --ntasks=1\n"
                     "#! #SBATCH --ntasks-per-core=1\n"
                     "#! #SBATCH --ntasks-per-node=2\n"
                     "#SBATCH --cpus-per-task=1\n"
                     "#SBATCH --time=05:00:00\n"
                     f"#SBATCH --array=1-{n_subjects:d}%{max_number_of_cpus:d}\n"
                     ". /home/opk20/miniconda3/bin/activate\n"
                     "conda activate fcest-env\n"
                     "conda info -e\n"
                     "export PYTHONPATH=$PWD:$PWD/GPflow\n"
                     "pwd\n"
                     f"srun --cpu_bind=threads --distribution=block:block python ./benchmarks/task/{data_set_name:s}/train_models/train_WP.py {model_name:s} {data_split:s} {repetition_time:s}")
    print('\n', slurm_job_str, '\n')

    # Create SLURM directory where logs will be saved (SLURM does not create these automatically).
    slurm_log_dir = os.path.join(cfg['experiments-basedir'], pp_pipeline, 'slurm_logs', data_split, model_name)
    if not os.path.exists(slurm_log_dir):
        os.makedirs(slurm_log_dir)

    slurm_job_filepath = save_slurm_job_to_file(
        config_dict=cfg,
        slurm_job_dir=os.path.join(
            cfg['experiments-basedir'], pp_pipeline, 'slurm_jobs', data_split, model_name
        ),
        slurm_job_str=slurm_job_str,
        data_split=data_split,
        model_name=model_name
    )
    slurm_job_id = send_slurm_job_to_queue(slurm_job_filepath)
    print(f"\n> SLURM job ID: {slurm_job_id}")
