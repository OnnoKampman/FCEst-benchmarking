import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.slurm import send_slurm_job_to_queue, save_slurm_job_to_file


if __name__ == "__main__":

    hostname = socket.gethostname()
    assert hostname == 'hivemind'

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # e.g. 'N0200_T0100'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    experiments_basedir = cfg['experiments-basedir']
    max_number_of_cpus = cfg['max-n-cpus']
    n_trials = int(experiment_data[-4:])

    dependency_line = ""
    for noise_type in cfg['noise-types']:
        slurm_job_str = ("#!/bin/bash\n"
                         f"#SBATCH --job-name=SIM_SAVE_{data_set_name:s}_{experiment_data:s}_{noise_type:s}_{data_split:s}\n"
                         "#! Output filename:\n"
                         f"#SBATCH --output={experiments_basedir:s}/slurm_logs/{noise_type:s}/{data_split:s}/SAVE/%A_%a.out\n"
                         "#! Errors filename:\n"
                         f"#SBATCH --error={experiments_basedir:s}/slurm_logs/{noise_type:s}/{data_split:s}/SAVE/%A_%a.err\n"
                         "#SBATCH --nodes=1\n"
                         "#SBATCH --ntasks=1\n"
                         "#! #SBATCH --ntasks-per-core=1\n"
                         "#! #SBATCH --ntasks-per-node=2\n"
                         "#SBATCH --cpus-per-task=1\n"
                         "#SBATCH --time=23:59:00\n"
                         f"#SBATCH --array=1-{n_trials:d}%{max_number_of_cpus:d}\n"
                         f"{dependency_line:s}"
                         ". /home/opk20/miniconda3/bin/activate\n"
                        #  "conda activate fcest-env\n"
                         "conda info -e\n"
                         "export PYTHONPATH=$PWD:$PWD/GPflow\n"
                         "pwd\n"
                         "echo \"SLURM_MEM_PER_CPU: $SLURM_MEM_PER_CPU\"\n"
                         "echo \"SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE\"\n"
                         "echo \"SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES\"\n"
                         "echo \"SLURM_NNODES: $SLURM_NNODES\"\n"
                         "echo \"SLURM_NTASKS: $SLURM_NTASKS\"\n"
                         "echo \"SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK\"\n"
                         "echo \"SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE\"\n"
                         f"srun --cpu_bind=threads --distribution=block:block python ./benchmarks/simulations/evaluation/compute_all_quantitative_results.py {data_set_name:s} {data_split:s} {experiment_data:s} {noise_type:s}")
        print('\n', slurm_job_str, '\n')

        # Create SLURM log directory (SLURM does not create these automatically).
        slurm_log_dir = os.path.join(
            cfg['experiments-basedir'], 'slurm_logs', noise_type, data_split, 'SAVE'
        )
        if not os.path.exists(slurm_log_dir):
            os.makedirs(slurm_log_dir)

        slurm_job_filepath = save_slurm_job_to_file(
            slurm_job_dir=os.path.join(
                cfg['experiments-basedir'], 'slurm_jobs', data_split, 'SAVE'
            ),
            slurm_job_str=slurm_job_str,
            slurm_job_filename=f'{noise_type:s}'
        )
        slurm_job_id = send_slurm_job_to_queue(slurm_job_filepath)
        print(f"\n> SLURM job ID: {slurm_job_id}")
        dependency_line = f"#SBATCH --dependency=afterany:{slurm_job_id:d}\n"
        print('\n\n\n')
