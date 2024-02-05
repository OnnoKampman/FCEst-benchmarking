import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split, get_tvfc_estimates
from helpers.rockland import get_rockland_subjects, load_rockland_data


if __name__ == "__main__":

    pp_pipeline = 'custom_fsl_pipeline'

    data_split = sys.argv[1]       # 'all', 'LEOO'
    metric = sys.argv[2]           # 'correlation', 'covariance'
    model_name = sys.argv[3]       # 'VWP_joint', 'SVWP_joint', 'SW_16', 'SW_30', 'SW_60', or 'sFC'
    repetition_time = sys.argv[4]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    n_subjects = len(all_subjects_list)

    tvfc_estimates_savedir = os.path.join(
        cfg['experiments-basedir'], pp_pipeline, 'TVFC_estimates', data_split, metric, model_name
    )
    if not os.path.exists(tvfc_estimates_savedir):
        os.makedirs(tvfc_estimates_savedir)

    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'\n> SUBJECT {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}\n')

        data_filepath = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_filepath)  # (N, 1), (N, D)

        match data_split:
            case 'LEOO':
                x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
            case _:
                x_train = x  # (N, 1)
                y_train = y  # (N, D)
        n_time_steps = x_train.shape[0]
        n_time_series = y_train.shape[1]

        estimated_tvfc = get_tvfc_estimates(
            config_dict=cfg,
            model_name=model_name,
            data_split=data_split,
            x_train=x_train,
            y_train=y_train,
            metric=metric,
            subject=subject_filename
        )  # (N, D, D)
        if estimated_tvfc is None:
            continue

        # Convert estimates to 2D array to save it to disk.
        estimated_tvfc_df = pd.DataFrame(estimated_tvfc.reshape(len(estimated_tvfc), -1).T)  # (D*D, N)

        estimated_tvfc_df.to_csv(os.path.join(tvfc_estimates_savedir, subject_filename))
        logging.info(f"Saved '{model_name:s}' TVFC estimates in '{tvfc_estimates_savedir:s}'.")
