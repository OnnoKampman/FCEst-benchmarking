import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split, get_test_log_likelihood, get_test_location_estimated_covariance_structure
from helpers.rockland import get_rockland_subjects, load_rockland_data


if __name__ == "__main__":

    model_name = sys.argv[1]  # 'VWP_joint', 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_16', 'SW_30', 'SW_60', or 'sFC'

    pp_pipeline = 'custom_fsl_pipeline'
    data_split = 'LEOO'  # leave-every-other-out

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)

    test_likelihoods_df = pd.DataFrame()
    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'\n> SUBJECT {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}\n')

        data_file = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        n_time_series = y.shape[1]

        x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
        y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)

        test_locations_predicted_covariance_structure = get_test_location_estimated_covariance_structure(
            config_dict=cfg,
            model_name=model_name,
            x_train_locations=x_train,
            x_test_locations=x_test,
            y_train_locations=y_train,
            data_split=data_split,
            subject=subject_filename,
            n_time_series=n_time_series
        )

        # Get likelihood of observed data at test locations under predicted covariance matrices.
        test_log_likelihood = get_test_log_likelihood(
            predicted_covariance_structure=test_locations_predicted_covariance_structure,
            y_test=y_test
        )
        test_likelihoods_df.loc[subject_filename.removesuffix('.csv'), 'test_likelihoods'] = test_log_likelihood

    print(test_likelihoods_df)
    likelihoods_filename = f'{data_split:s}_likelihoods_{model_name:s}.csv'
    test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
    if not os.path.exists(test_likelihoods_savedir):
        os.makedirs(test_likelihoods_savedir)
    test_likelihoods_df.to_csv(
        os.path.join(test_likelihoods_savedir, likelihoods_filename),
        index=True,
        float_format='%.2f'
    )
    logging.info(f"Saved {data_split:s} likelihoods '{likelihoods_filename:s}' in '{test_likelihoods_savedir:s}'.")
