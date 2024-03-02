import logging
import os
import socket
import sys

from fcest.models.sliding_windows import SlidingWindows
import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.rockland import get_rockland_subjects, load_rockland_data


if __name__ == "__main__":

    model_name = 'SW_cross_validated'
    pp_pipeline = 'custom_fsl_pipeline'

    data_split = sys.argv[1]       # 'all' or 'LEOO'
    repetition_time = sys.argv[2]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    n_subjects = len(all_subjects_list)
    repetition_time_seconds = float(repetition_time) / 1000

    # Allow for local and CPU cluster training.
    if socket.gethostname() == 'hivemind':
        try:
            i_subject = os.environ['SLURM_ARRAY_TASK_ID']
            i_subject = int(i_subject) - 1  # to make zero-index
            print('SLURM trial ID', i_subject)
            subjects = [
                all_subjects_list[i_subject]
            ]
        except KeyError:
            subjects = all_subjects_list
    else:
        subjects = all_subjects_list

    optimal_window_lengths_df = pd.DataFrame()
    for i_subject, subject_filename in enumerate(subjects):
        print(f'\n> SUBJECT {i_subject+1:d} / {n_subjects:d}: {subject_filename:s}\n')

        data_file = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        if data_split == 'LEOO':
            x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), _
            y_train, _ = leave_every_other_out_split(y)  # (N/2, 1), _
        else:
            x_train = x  # (N, 1)
            y_train = y  # (N, D)
        n_time_series = y_train.shape[1]

        m = SlidingWindows(
            x_train_locations=x_train,
            y_train_locations=y_train,
            repetition_time=repetition_time_seconds
        )
        optimal_window_length = m.compute_cross_validated_optimal_window_length()
        optimal_window_lengths_df.loc[subject_filename.removesuffix('.csv'), 'cross_validated_window_length'] = int(optimal_window_length)

        # Save model estimates (both covariances and correlations).
        for metric in ['correlation', 'covariance']:
            model_estimates_savedir = os.path.join(
                cfg['experiments-basedir'], pp_pipeline, 'TVFC_estimates', data_split, metric, model_name
            )
            m.save_tvfc_estimates(
                optimal_window_length=optimal_window_length,
                savedir=model_estimates_savedir,
                model_name=subject_filename,
                connectivity_metric=metric
            )

    savedir = os.path.join(cfg['git-results-basedir'], 'optimal_window_lengths', data_split)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    optimal_window_lengths_df.to_csv(
        os.path.join(savedir, 'optimal_window_lengths.csv')
    )
    logging.info(f"Saved {data_split:s} optimal window lengths in git repository.")
