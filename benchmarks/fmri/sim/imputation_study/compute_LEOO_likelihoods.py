import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split, get_test_log_likelihood, get_test_location_estimated_covariance_structure


if __name__ == "__main__":

    data_split = 'LEOO'  # leave-every-other-out

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    experiment_data = sys.argv[2]  # e.g. 'N0200_T0200'
    model_name = sys.argv[3]       # 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_{%d}', 'sFC'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    num_trials = int(experiment_data[-4:])

    for noise_type in cfg['noise-types']:

        test_likelihoods_df = pd.DataFrame(
            index=range(num_trials),
            columns=cfg['all-covs-types'],
        )
        for i_trial in range(num_trials):

            for covs_type in cfg['all-covs-types']:
                data_filepath = os.path.join(
                    cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                    f'{covs_type:s}_covariance.csv'
                )
                if not os.path.exists(data_filepath):
                    logging.warning(f"File '{data_filepath:s}' not found.")
                    if covs_type == 'boxcar':
                        data_filepath = os.path.join(
                            cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                            'checkerboard_covariance.csv'
                        )
                        if not os.path.exists(data_filepath):
                            logging.warning(f"File '{data_filepath:s}' not found.")
                            continue
                x, y = load_data(
                    data_filepath,
                    verbose=False,
                )  # (N, 1), (N, D)
                n_time_series = y.shape[1]  # D

                x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)

                test_locations_predicted_covariance_structure = get_test_location_estimated_covariance_structure(
                    config_dict=cfg,
                    model_name=model_name,
                    n_time_series=n_time_series,
                    x_train_locations=x_train,
                    y_train_locations=y_train,  # only used for VWP models
                    x_test_locations=x_test,
                    data_split=data_split,
                    i_trial=i_trial,
                    noise_type=noise_type,
                    covs_type=covs_type,
                    subject='',
                    connectivity_metric='covariance',
                )  # (N_test, D, D)

                # Get likelihood of observed data at test locations under predicted covariance matrices.
                test_log_likelihood = get_test_log_likelihood(
                    predicted_covariance_structure=test_locations_predicted_covariance_structure,
                    y_test=y_test
                )
                test_likelihoods_df.loc[i_trial, covs_type] = test_log_likelihood

        test_likelihoods_df = test_likelihoods_df.round(2)
        print(test_likelihoods_df.astype(float).round(2))

        likelihoods_filename = f'{data_split:s}_{noise_type:s}_likelihoods_{model_name:s}.csv'
        test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
        if not os.path.exists(test_likelihoods_savedir):
            os.makedirs(test_likelihoods_savedir)
        test_likelihoods_df.astype(float).to_csv(
            os.path.join(test_likelihoods_savedir, likelihoods_filename),
            index=True,
            float_format='%.2f'
        )
        logging.info(f"Saved {data_split:s} likelihoods '{likelihoods_filename:s}' in '{test_likelihoods_savedir:s}'.")
