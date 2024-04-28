import logging
import os
import socket
import sys

import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data
from helpers.evaluation import leave_every_other_out_split, get_test_log_likelihood, get_test_location_estimated_covariance_structure


if __name__ == "__main__":

    data_split = 'LEOO'  # leave-every-other-out
    experiment_dimensionality = 'multivariate'  # or 'bivariate'

    data_dimensionality = sys.argv[1]        # 'd15', 'd50'
    model_name = sys.argv[2]                 # 'SVWP_joint', 'SW_30', 'SW_60', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'GO_joint', 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    num_subjects = cfg['n-subjects']
    all_subjects_filenames_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=num_subjects,
    )

    test_log_likelihoods_df = pd.DataFrame(
        np.nan,
        index=all_subjects_filenames_list,
        columns=cfg['scan-ids'],
    )
    for i_subject, subject_filename in enumerate(all_subjects_filenames_list):

        logging.info(f'> SUBJECT {i_subject+1: 3d} / {num_subjects:d}: {subject_filename:s}')

        data_file = os.path.join(
            cfg['data-dir'], subject_filename
        )
        for scan_id in cfg['scan-ids']:
            x, y = load_human_connectome_project_data(
                data_file,
                scan_id=scan_id,
                verbose=False,
            )
            n_time_series = y.shape[1]
            if experiment_dimensionality == 'bivariate':
                chosen_indices = [0, 1]
                # chosen_indices_df = cfg['chosen-indices']
                # chosen_indices = chosen_indices_df.loc[subject, scan_id]
                y = y[:, chosen_indices]
                n_time_series = y.shape[1]
                print('y', y.shape)

            x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
            y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)

            # Get estimated covariance structure at test locations.
            test_locations_predicted_covariance_structure = get_test_location_estimated_covariance_structure(
                config_dict=cfg,
                model_name=model_name,
                n_time_series=n_time_series,
                x_train_locations=x_train,
                x_test_locations=x_test,
                scan_id=scan_id,
                data_split=data_split,
                experiment_dimensionality=experiment_dimensionality,
                subject=subject_filename,
            )

            # Get likelihood of observed data at test locations under predicted covariance matrices.
            if test_locations_predicted_covariance_structure is not None:
                test_log_likelihood = get_test_log_likelihood(
                    predicted_covariance_structure=test_locations_predicted_covariance_structure,
                    y_test=y_test
                )
                test_log_likelihoods_df.loc[subject_filename, scan_id] = test_log_likelihood

    print(test_log_likelihoods_df)
    filename = f'{data_split}_{experiment_dimensionality:s}_likelihoods_{model_name:s}.csv'
    savedir = os.path.join(
        cfg['git-results-basedir'], 'imputation_study'
    )
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    test_log_likelihoods_df.astype(float).to_csv(
        os.path.join(savedir, filename),
        header=True,
        index=True,
        float_format='%.2f'
    )
    logging.info(f"Saved {data_split:s} likelihoods '{filename:s}' in '{savedir:s}'.")
