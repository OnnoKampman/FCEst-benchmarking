import logging
import os
import socket
import sys

import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import get_corr_matrices_tril_spearman_correlation
from helpers.evaluation import get_cov_matrices_rmse, get_corr_matrices_rmse, leave_every_other_out_split
from helpers.evaluation import get_d2_covariance_term_correlation, get_test_log_likelihood
from helpers.evaluation import get_d2_covariance_term_rmse, get_d2_correlation_term_rmse
from helpers.evaluation import get_tvfc_estimates
from helpers.synthetic_covariance_structures import get_ground_truth_covariance_structure


def _get_performance_metric(
    performance_metric: str,
    predicted_covariance_structure_test_locations: np.array,
    ground_truth_covariance_structure_test_locations: np.array,
    y_test_locations: np.array,
) -> float:
    """
    Computes performance metric.

    Parameters
    ----------
    :param performance_metric:
    :param predicted_covariance_structure_test_locations: np.array,
    :param ground_truth_covariance_structure_test_locations:
    :param y_test_locations:
    :return:
    """
    match performance_metric:
        case 'covariance_RMSE':
            return get_d2_covariance_term_rmse(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'correlation_RMSE':
            return get_d2_correlation_term_rmse(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'covariance_correlation':
            return get_d2_covariance_term_correlation(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'covariance_matrix_RMSE':
            return get_cov_matrices_rmse(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'correlation_matrix_RMSE':
            return get_corr_matrices_rmse(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'correlation_tril_matrix_elements_RMSE':
            raise NotImplementedError
        case 'covariance_tril_matrix_elements_RMSE':
            raise NotImplementedError
        case 'tril_correlation':
            return get_corr_matrices_tril_spearman_correlation(
                predicted_covariance_structure_test_locations, ground_truth_covariance_structure_test_locations
            )
        case 'test_log_likelihood':
            return get_test_log_likelihood(predicted_covariance_structure_test_locations, y_test_locations)
        case _:
            logging.error(f"Performance metric '{performance_metric:s}' not recognized.")


if __name__ == "__main__":

    hostname = socket.gethostname()

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # e.g. 'N0200_T0200'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    models_list = cfg['all-quantitative-results-models']
    n_trials = int(experiment_data[-4:])

    if hostname == 'hivemind':
        if len(sys.argv) == 5:
            noise_types = [sys.argv[4]]
            i_trials = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]  # to make zero-index
        else:
            noise_types = cfg['noise-types']
            i_trials = range(n_trials)
    else:
        print('Running locally...')
        noise_types = cfg['noise-types']
        i_trials = range(n_trials)

    for noise_type in noise_types:
        if noise_type != 'no_noise':
            SNR = float(noise_type[-1])
            print('SNR', SNR)
        else:
            SNR = None
        for i_trial in i_trials:
            for perform_metric in cfg['performance-metrics']:
                performance_df = pd.DataFrame(
                    index=models_list,
                    columns=cfg['all-covs-types']
                )
                for covs_type in cfg['all-covs-types']:

                    data_file = os.path.join(
                        cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                        f'{covs_type:s}_covariance.csv'
                    )
                    if not os.path.exists(data_file):

                        # Fix renaming issue.
                        if covs_type == 'boxcar':
                            covs_type = 'checkerboard'
                            data_file = os.path.join(
                                cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                                f'{covs_type:s}_covariance.csv'
                            )

                        if not os.path.exists(data_file):
                            logging.warning(f"Data file '{data_file:s}' not found.")
                            performance_df.loc[:, covs_type] = np.nan
                            continue

                    x, y = load_data(
                        data_file,
                        verbose=False,
                    )  # (N, 1), (N, D)
                    n_time_series = y.shape[1]  # D

                    gt_covariance_structure = get_ground_truth_covariance_structure(
                        covs_type=covs_type,
                        n_samples=len(x),
                        signal_to_noise_ratio=SNR,
                        data_set_name=data_set_name
                    )

                    # Select train and test data through a leave-every-other-out (LEOO) split.
                    if data_split == 'LEOO':
                        x_train, x_test = leave_every_other_out_split(x)
                        y_train, y_test = leave_every_other_out_split(y)
                        gt_covariance_structure_train, gt_covariance_structure_test = leave_every_other_out_split(gt_covariance_structure)
                    else:
                        x_train = x_test = x
                        y_train = y_test = y
                        gt_covariance_structure_train = gt_covariance_structure_test = gt_covariance_structure

                    for tvfc_estimation_method in models_list:
                        estimated_cov_structure_test = get_tvfc_estimates(
                            config_dict=cfg,
                            model_name=tvfc_estimation_method,
                            x_train=x_train,
                            y_train=y_train,
                            noise_type=noise_type,
                            i_trial=i_trial,
                            covs_type=covs_type,
                            data_split=data_split,
                            metric='covariance'
                        )  # (N_test, D, D)
                        if estimated_cov_structure_test is None:
                            logging.warning(f"Estimates for '{tvfc_estimation_method:s}' not found.")
                            performance_df.loc[tvfc_estimation_method, covs_type] = np.nan
                            continue
                        performance_df.loc[tvfc_estimation_method, covs_type] = _get_performance_metric(
                            performance_metric=perform_metric,
                            predicted_covariance_structure_test_locations=estimated_cov_structure_test,
                            ground_truth_covariance_structure_test_locations=gt_covariance_structure_test,
                            y_test_locations=y_test
                        )
                performance_df = performance_df.round(4)
                print(performance_df.round(4))

                quantitative_results_savedir = os.path.join(
                    cfg['experiments-basedir'], noise_type, data_split,
                    f'trial_{i_trial:03d}'
                )
                if not os.path.exists(quantitative_results_savedir):
                    os.makedirs(quantitative_results_savedir)
                quantitative_results_savepath = os.path.join(
                    quantitative_results_savedir, f'{perform_metric:s}.csv'
                )
                performance_df.to_csv(
                    quantitative_results_savepath,
                    index=True
                )
                logging.info(f"Saved quantitative results in '{quantitative_results_savedir:s}'.")
