import logging
import os
import socket
import sys

from fcest.models.sliding_windows import SlidingWindows
import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    model_name = 'SW_cross_validated'

    data_dimensionality = sys.argv[1]        # 'd15' or 'd50'
    data_split = sys.argv[2]                 # 'all' or 'LEOO'
    experiment_dimensionality = sys.argv[3]  # 'multivariate' or 'bivariate'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=hostname
    )
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir']
    )
    num_subjects = cfg['n-subjects']

    # Allow for local and CPU cluster training.
    # When running on the Hivemind with SLURM, only one model is trained here.
    if hostname == 'hivemind':
        try:
            i_subject = os.environ['SLURM_ARRAY_TASK_ID']
            i_subject = int(i_subject) - 1  # to make zero-index
            print('SLURM trial ID', i_subject)
            subjects = [
                all_subjects_list[i_subject]
            ]
        except KeyError:
            subjects = all_subjects_list[:num_subjects]
    else:
        print('Running locally...')
        subjects = all_subjects_list[:num_subjects]

    optimal_window_lengths_df = pd.DataFrame()
    for i_subject, subject_filename in enumerate(subjects):

        print(f'\n> SUBJECT {i_subject+1:d} / {num_subjects:d}: {subject_filename:s}')

        data_file = os.path.join(cfg['data-dir'], subject_filename)
        for scan_id in cfg['scan-ids']:
            print(f'\nSCAN {scan_id+1:d} / 4\n')

            x, y = load_human_connectome_project_data(
                data_file,
                scan_id=scan_id,
                verbose=False,
            )  # (N, 1), (N, D)

            match data_split:
                case "LEOO":
                    x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                    y_train, _ = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
                case "all":
                    x_train = x  # (N, 1)
                    y_train = y  # (N, D)
                case _:
                    logging.error("Data split not recognized.")
                    continue

            m = SlidingWindows(
                x_train_locations=x_train,
                y_train_locations=y_train,
                repetition_time=cfg['repetition-time'],
            )
            optimal_window_length = m.compute_cross_validated_optimal_window_length()

            # TODO: this only works when not running it in parallel!
            optimal_window_lengths_df.loc[subject_filename.removesuffix('.txt'), scan_id] = optimal_window_length

            for metric in ['correlation', 'covariance']:
                model_estimates_savedir = os.path.join(
                    cfg['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                    data_split, experiment_dimensionality, metric, model_name
                )
                m.save_tvfc_estimates(
                    optimal_window_length=optimal_window_length,
                    savedir=model_estimates_savedir,
                    model_name=f"{subject_filename.removesuffix('.txt'):s}.csv",
                    connectivity_metric=metric
                )

    savedir = os.path.join(cfg['git-results-basedir'], 'optimal_window_lengths', data_split)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    optimal_window_lengths_df.to_csv(
        os.path.join(savedir, 'optimal_window_lengths.csv')
    )
    logging.info(f"Saved {data_split:s} optimal window lengths in git repository.")
