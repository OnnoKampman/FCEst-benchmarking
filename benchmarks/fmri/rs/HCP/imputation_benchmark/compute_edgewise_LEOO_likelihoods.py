import logging
import os
import socket
import sys

from fcest.helpers.array_operations import get_all_lower_triangular_indices_tuples
import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.array_operations import slice_covariance_structure
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data
from helpers.evaluation import leave_every_other_out_split, get_test_log_likelihood, get_test_location_estimated_covariance_structure


if __name__ == "__main__":

    data_split = 'LEOO'  # leave-every-other-out
    experiment_dimensionality = 'multivariate'  # TODO: take all edges from the jointly/multivariately trained model

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    model_name = sys.argv[2]           # 'SVWP_joint', 'DCC_joint', 'GO_joint', 'SW_cross_validated', 'SW_30', 'SW_60', 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    num_subjects = cfg['n-subjects']
    num_time_series = int(data_dimensionality[1:])
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'], first_n_subjects=num_subjects
    )

    edgewise_LEOO_likelihoods_df = pd.DataFrame(
        np.zeros((num_time_series, num_time_series))
    )  # (D, D)
    num_estimates = 0
    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'\n> SUBJECT {i_subject+1:d} / {num_subjects:d}: {subject_filename:s}')

        data_file = os.path.join(
            cfg['data-dir'], subject_filename
        )
        for scan_id in cfg['scan-ids']:

            x, y = load_human_connectome_project_data(
                data_file,
                scan_id=scan_id,
                verbose=False,
            )  # (N, 1), (N, D)
            num_time_series = y.shape[1]

            x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
            y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)

            # Get full covariance structure at test locations.
            test_locations_predicted_covariance_structure = get_test_location_estimated_covariance_structure(
                config_dict=cfg,
                model_name=model_name,
                num_time_series=num_time_series,
                x_train_locations=x_train,
                x_test_locations=x_test,
                scan_id=scan_id,
                data_split=data_split,
                experiment_dimensionality=experiment_dimensionality,
                subject=subject_filename,
            )  # (N_test, D, D)

            if test_locations_predicted_covariance_structure is None:
                continue

            subject_scan_test_likelihoods_edgewise_array = np.zeros((num_time_series, num_time_series))  # (D, D)
            for edge_tuple in get_all_lower_triangular_indices_tuples(num_time_series):
                y_test_edge = y_test[:, edge_tuple]  # (N_test, 2)
                num_time_series = y_test_edge.shape[1]

                test_locations_predicted_covariance_structure_edge = slice_covariance_structure(
                    test_locations_predicted_covariance_structure, edge_tuple
                )  # (N_test, 2, 2)

                # Get likelihood of observed data at test locations under predicted covariance matrices.
                test_log_likelihood = get_test_log_likelihood(
                    predicted_covariance_structure=test_locations_predicted_covariance_structure_edge,
                    y_test=y_test_edge
                )

                # Fill in subject/scan edgewise likelihoods.
                subject_scan_test_likelihoods_edgewise_array[edge_tuple] = test_log_likelihood
                subject_scan_test_likelihoods_edgewise_array[(edge_tuple[1], edge_tuple[0])] = test_log_likelihood

            edgewise_LEOO_likelihoods_df += subject_scan_test_likelihoods_edgewise_array
            num_estimates += 1

    # Take the average over subjects and scans.
    edgewise_LEOO_likelihoods_df /= num_estimates

    print(edgewise_LEOO_likelihoods_df)
    filename = f'{data_split}_{experiment_dimensionality:s}_likelihoods_{model_name:s}_edgewise.csv'
    savedir = os.path.join(
        cfg['git-results-basedir'], 'imputation_benchmark'
    )
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    edgewise_LEOO_likelihoods_df.to_csv(
        os.path.join(savedir, filename),
        index=True,
        float_format='%.2f'
    )
    logging.info(f"Saved edgewise {data_split:s} likelihoods '{filename:s}' in '{savedir:s}'.")
