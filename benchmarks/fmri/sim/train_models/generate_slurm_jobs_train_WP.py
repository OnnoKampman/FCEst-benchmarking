import logging
import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.slurm import send_slurm_job_to_queue


if __name__ == "__main__":

    hostname = socket.gethostname()
    assert hostname == 'hivemind'
    assert len(sys.argv) == 5

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # e.g. 'N0200_T0100'
    model_name = sys.argv[4]       # 'VWP_joint', 'SVWP_joint', 'VWP', or 'SVWP'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    benchmark_basedir = cfg['experiments-basedir']
    max_number_of_cpus = cfg['max-n-cpus']
    n_trials = int(experiment_data[-4:])

    experiment_dir = os.path.join(
        cfg['experiments-basedir'], 'slurm_jobs', data_split, model_name
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    dependency_line = ""
    for noise_type in cfg['noise-types']:
        for covs_type in cfg['all-covs-types']:
            slurm_job_str = ("#!/bin/bash\n"
                             f"#SBATCH --job-name=SIM_{data_set_name:s}_{data_split:s}_{model_name:s}_{experiment_data:s}_{noise_type:s}_{covs_type:s}\n"
                             "#! Output filename:\n"
                             "#! %A means slurm job ID and %a means array index\n"
                             f"#SBATCH --output={benchmark_basedir:s}/slurm_logs/{noise_type:s}/{covs_type:s}/{data_split:s}/{model_name:s}/%A_%a.out\n"
                             "#! Errors filename:\n"
                             f"#SBATCH --error={benchmark_basedir:s}/slurm_logs/{noise_type:s}/{covs_type:s}/{data_split:s}/{model_name:s}/%A_%a.err\n"
                             "#! Number of nodes to be allocated for the job (for single core jobs always leave this at 1)\n"
                             "#SBATCH --nodes=1\n"
                             "#! Number of tasks. By default SLURM assumes 1 task per node and 1 CPU per task. (for single core jobs always leave this at 1)\n"
                             "#SBATCH --ntasks=1\n"
                             "#! #SBATCH --ntasks-per-core=1\n"
                             "#! #SBATCH --ntasks-per-node=2\n"
                             "#! How many many cores will be allocated per task? (for single core jobs always leave this at 1)\n"
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
                             f"srun --cpu_bind=threads --distribution=block:block python ./benchmarks/fmri/sim/train_models/train_WP.py {data_set_name:s} {experiment_data:s} {model_name:s} {data_split:s} {noise_type:s} {covs_type:s}")
            print('\n', slurm_job_str, '\n')

            # Create SLURM log directory (SLURM does not create these automatically).
            slurm_log_dir = os.path.join(
                cfg['experiments-basedir'], 'slurm_logs', noise_type, covs_type, data_split, model_name
            )
            if not os.path.exists(slurm_log_dir):
                os.makedirs(slurm_log_dir)

            slurm_job_filepath = os.path.join(experiment_dir, f'{noise_type:s}_{covs_type:s}')
            with open(slurm_job_filepath, 'w') as f:
                f.write(slurm_job_str)
            logging.info(f"Saved SLURM job '{slurm_job_filepath:s}'.")

            slurm_job_id = send_slurm_job_to_queue(slurm_job_filepath)
            print(f"\n> SLURM job ID: {slurm_job_id}")
            dependency_line = f"#SBATCH --dependency=afterany:{slurm_job_id:d}\n"
            print('\n\n\n')
