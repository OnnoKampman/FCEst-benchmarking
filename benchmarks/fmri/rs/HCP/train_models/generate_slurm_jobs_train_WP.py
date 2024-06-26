import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects
from helpers.slurm import save_slurm_job_to_file, send_slurm_job_to_queue


if __name__ == "__main__":

    hostname = socket.gethostname()
    assert hostname == 'hivemind'

    experiment_dimensionality = 'multivariate'  # or 'bivariate'
    model_name = 'SVWP_joint'

    data_dimensionality = sys.argv[1]        # 'd15' or 'd50'
    data_split = sys.argv[2]                 # 'all' or 'LEOO'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=hostname
    )
    data_set_name = cfg['data-set-name']
    experiments_basedir = cfg['experiments-basedir']
    max_number_of_cpus = cfg['max-n-cpus']

    n_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'], first_n_subjects=n_subjects
    )

    slurm_job_str = ("#!/bin/bash\n"
                     f"#SBATCH --job-name=HCP_{data_split:s}_{model_name:s}\n"
                     "#! Output filename:\n"
                     f"#SBATCH --output={experiments_basedir:s}/slurm_logs/{data_split:s}/{model_name:s}/%A_%a.out\n"
                     "#! Errors filename:\n"
                     f"#SBATCH --error={experiments_basedir:s}/slurm_logs/{data_split:s}/{model_name:s}/%A_%a.err\n"
                     "#SBATCH --nodes=1\n"
                     "#SBATCH --ntasks=1\n"
                     "#! #SBATCH --ntasks-per-core=1\n"
                     "#! #SBATCH --ntasks-per-node=2\n"
                     "#SBATCH --cpus-per-task=1\n"
                     "#SBATCH --time=96:00:00\n"
                     f"#SBATCH --array=1-{n_subjects:d}%{max_number_of_cpus:d}\n"
                     ". /home/opk20/miniconda3/bin/activate\n"
                     "conda activate fcest-env\n"
                     "conda info -e\n"
                     "export PYTHONPATH=$PWD:$PWD/GPflow\n"
                     "export LD_LIBRARY_PATH=/home/opk20/miniconda3/lib\n"
                     "pwd\n"
                     "echo \"SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU\"\n"
                     "echo \"SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE\"\n"
                     "echo \"SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES\"\n"
                     "echo \"SLURM_NNODES: $SLURM_NNODES\"\n"
                     "echo \"SLURM_NTASKS: $SLURM_NTASKS\"\n"
                     "echo \"SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK\"\n"
                     "echo \"SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE\"\n"
                     f"srun --cpu_bind=threads --distribution=block:block python ./benchmarks/fmri/rs/{data_set_name:s}/train_models/train_SVWP.py {data_dimensionality:s} {data_split:s}")
    print('\n', slurm_job_str, '\n')

    # Create SLURM directory where logs will be saved (SLURM does not create these automatically).
    slurm_log_dir = os.path.join(cfg['experiments-basedir'], 'slurm_logs', data_split, model_name)
    if not os.path.exists(slurm_log_dir):
        os.makedirs(slurm_log_dir)

    slurm_job_filepath = save_slurm_job_to_file(
        slurm_job_dir=os.path.join(
            cfg['experiments-basedir'], 'slurm_jobs', data_split, model_name
        ),
        slurm_job_str=slurm_job_str
    )
    slurm_job_id = send_slurm_job_to_queue(slurm_job_filepath)
    print(f"\n> SLURM job ID: {slurm_job_id}")
