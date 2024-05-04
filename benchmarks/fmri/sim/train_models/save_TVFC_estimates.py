import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split, get_tvfc_estimates


if __name__ == "__main__":

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    data_split = sys.argv[2]       # 'all' or 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'
    metric = sys.argv[4]           # 'correlation' or 'covariance'
    model_name = sys.argv[5]       # 'VWP', 'VWP_joint', 'SVWP', 'SVWP_joint', 'SW_15', 'SW_30', 'SW_60', 'SW_120', or 'sFC'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    num_trials = int(experiment_data[-4:])

    i_trials = range(num_trials)
    for noise_type in cfg['noise-types']:

        for i_trial in i_trials:

            for covs_type in cfg['all-covs-types']:

                data_filepath = os.path.join(
                    cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                    f'{covs_type:s}_covariance.csv'
                )
                if not os.path.exists(data_filepath):
                    logging.warning(f"Node time series not found: '{data_filepath:s}'")
                    if covs_type == 'boxcar':
                        data_filepath = os.path.join(
                            cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                            'checkerboard_covariance.csv'
                        )
                    if not os.path.exists(data_filepath):
                        logging.warning(f"Node time series not found: '{data_filepath:s}'")
                        continue
                    continue

                tvfc_estimates_savedir = os.path.join(
                    cfg['experiments-basedir'], noise_type, f'trial_{i_trial:03d}', 'TVFC_estimates',
                    data_split, metric, model_name
                )
                tvfc_estimates_savefilepath = os.path.join(
                    tvfc_estimates_savedir, f'{covs_type:s}.csv'
                )
                if os.path.exists(tvfc_estimates_savefilepath):
                    logging.info(f"Found existing TVFC estimates in '{tvfc_estimates_savefilepath:s}'.")
                    continue
                if not os.path.exists(tvfc_estimates_savedir):
                    os.makedirs(tvfc_estimates_savedir)
                x, y = load_data(
                    data_filepath,
                    verbose=False,
                )  # (N, 1), (N, D)
                num_time_steps = x.shape[0]
                num_time_series = y.shape[1]

                match data_split:
                    case "LEOO":
                        x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                        y_train, _ = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
                    case "all":
                        x_train = x  # (N, 1)
                        y_train = y  # (N, D)
                    case _:
                        logging.error(f"Data split '{data_split:s}' not recognized.")

                estimated_tvfc = get_tvfc_estimates(
                    config_dict=cfg,
                    model_name=model_name,
                    x_train=x_train,
                    y_train=y_train,
                    metric=metric,
                    data_split=data_split,
                    noise_type=noise_type,
                    i_trial=i_trial,
                    covs_type=covs_type
                )
                if estimated_tvfc is None:
                    continue

                # Convert estimates to 2D array to save it to disk.
                estimated_tvfc_df = pd.DataFrame(estimated_tvfc.reshape(len(estimated_tvfc), -1).T)  # (D*D, N)

                estimated_tvfc_df.to_csv(tvfc_estimates_savefilepath)
                logging.info(f"Saved {model_name:s} TVFC estimates in '{tvfc_estimates_savedir:s}'.")
