import logging
import os
import re
import subprocess


def send_slurm_job_to_queue(slurm_job_filepath: str):
    """
    Run SLURM job.
    """
    output = subprocess.run(
        ["sbatch", f"{slurm_job_filepath:s}"],
        capture_output=True
    )
    print('\n', output, '\n')

    output_str = output.stdout.decode("utf-8")

    slurm_job_id = int(re.match('.*?([0-9]+)$', output_str).group(1))  # get digits at the end of string

    return slurm_job_id


def save_slurm_job_to_file(
        slurm_job_dir: str, slurm_job_str: str, slurm_job_filename: str = None, config_dict: dict = None, data_split: str = None, model_name: str = None
) -> str:
    """
    Save SLURM job definition to file, which can then be submitted and executed.

    :param slurm_job_dir:
    :param slurm_job_str:
    :param slurm_job_filename:
    :param config_dict:
    :param data_split:
    :param model_name:
    :return:
    """
    if slurm_job_filename is None:
        slurm_job_filename = 'all_subjects'
    if not os.path.exists(slurm_job_dir):
        os.makedirs(slurm_job_dir)
    slurm_job_filepath = os.path.join(slurm_job_dir, slurm_job_filename)
    with open(slurm_job_filepath, 'w') as f:
        f.write(slurm_job_str)
    logging.info(f"Saved SLURM job '{slurm_job_filepath:s}'.")
    return slurm_job_filepath
