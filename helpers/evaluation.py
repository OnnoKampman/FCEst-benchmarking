import logging
import os

from fcest.models.sliding_windows import SlidingWindows
from fcest.models.wishart_process import SparseVariationalWishartProcess, VariationalWishartProcess
from fcest.helpers.array_operations import to_correlation_structure
from fcest.helpers.data import to_3d_format
import gpflow
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, spearmanr


def get_d2_covariance_term_rmse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    """
    Here we evaluate only the covariance term for the bivariate (D = 2) case.

    :param predicted_covariance_structure: (N_test, D, D)
    :param ground_truth_covariance_structure: (N_test, D, D)
    :return:
        root mean squared error between estimated covariance term and ground truth covariance term.
    """
    return np.sqrt(
        _get_d2_covariance_term_mse(predicted_covariance_structure, ground_truth_covariance_structure)
    )


def _get_d2_covariance_term_mse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    """
    Here we evaluate only the covariance term for the bivariate (D = 2) case.

    :param predicted_covariance_structure: (N_test, D, D)
    :param ground_truth_covariance_structure: (N_test, D, D)
    :return:
        mean squared error between estimated covariance term and ground truth covariance term.
    """
    predicted_covariance_terms = predicted_covariance_structure[:, 0, 1]
    ground_truth_covariance_terms = ground_truth_covariance_structure[:, 0, 1]
    return np.mean((predicted_covariance_terms - ground_truth_covariance_terms) ** 2)


def get_d2_correlation_term_rmse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    """
    Here we evaluate only the correlation term for the bivariate (D = 2) case.

    :param predicted_covariance_structure: (N_test, D, D)
    :param ground_truth_covariance_structure: (N_test, D, D)
    :return:
    """
    predicted_correlation_structure = to_correlation_structure(predicted_covariance_structure)
    ground_truth_correlation_structure = to_correlation_structure(ground_truth_covariance_structure)
    return get_d2_covariance_term_rmse(
        predicted_correlation_structure, ground_truth_correlation_structure
    )


def get_d2_covariance_term_correlation(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    """
    TODO: this doesn't work for static ones (cannot compute std for that one)

    :param predicted_covariance_structure: (N_test, D, D)
    :param ground_truth_covariance_structure: (N_test, D, D)
    :return:
    """
    predicted_covariance_terms = predicted_covariance_structure[:, 0, 1]
    ground_truth_covariance_terms = ground_truth_covariance_structure[:, 0, 1]
    return np.corrcoef(
        predicted_covariance_terms, ground_truth_covariance_terms
    )[0, 1]


def get_cov_matrices_rmse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    return np.sqrt(
        _get_cov_matrices_mse(predicted_covariance_structure, ground_truth_covariance_structure)
    )


def get_corr_matrices_rmse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    predicted_correlation_structure = to_correlation_structure(predicted_covariance_structure)
    ground_truth_correlation_structure = to_correlation_structure(ground_truth_covariance_structure)
    return get_cov_matrices_rmse(
        predicted_correlation_structure, ground_truth_correlation_structure
    )


def _get_cov_matrices_mse(
        predicted_covariance_structure: np.array, ground_truth_covariance_structure: np.array
) -> float:
    """
    We evaluate on the whole covariance matrix, so that all off-diagonal values are accounted for twice.
    This is what is done in the original Generalized Wishart Processes paper as well.

    :param predicted_covariance_structure: (N_test, D, D)
    :param ground_truth_covariance_structure: (N_test, D, D)
    :return:
    """
    return np.mean((predicted_covariance_structure - ground_truth_covariance_structure) ** 2)


def get_covariance_mse(predicted_covs, covs_test, x_train, x_test):
    """
    We use interpolation to get values in between windows.
    We compute the mean over the element-wise squared error of the entire covariance matrix.

    :param predicted_covs: shape of (N, D, D)
    """
    n_time_series = predicted_covs.shape[1]
    full_cov_estimation_test_locations = np.zeros_like(predicted_covs)
    for i in range(n_time_series):
        for j in range(i, n_time_series):
            interpolated_covariances = np.interp(
                x=x_test,
                xp=x_train[:, 0],
                fp=predicted_covs[:, i, j]
            )
            full_cov_estimation_test_locations[:, i, j] = interpolated_covariances[:, 0]
            full_cov_estimation_test_locations[:, j, i] = interpolated_covariances[:, 0]
    return _get_cov_matrices_mse(full_cov_estimation_test_locations, covs_test)


def get_corr_matrices_tril_spearman_correlation():
    """
    TODO: extract lower triangular values and then run Spearman correlation on the two resulting vectors
    :return:
    """
    c = spearmanr()
    raise NotImplementedError


def get_test_log_likelihood(predicted_covariance_structure: np.array, y_test: np.array) -> float:
    """
    Evaluate log likelihood per data point (average instead of sum) under a zero mean Gaussian distribution.
    We use interpolation to get the covariance matrices in between train locations.
    This needs to be done before feeding into this function.

    :param predicted_covariance_structure: shape of (N_test, D, D)
    :param y_test: (N_test, D)
    :return: float value: mean likelihood over test locations
    """
    n_time_series = predicted_covariance_structure.shape[1]  # D

    all_observation_log_likelihoods = []
    for test_data_point_index in range(len(y_test)):
        eval_location_estimated_covariance_matrix = predicted_covariance_structure[test_data_point_index]
        
        # if not is_positive_definite(eval_location_estimated_covariance_matrix):
        #     eval_location_estimated_covariance_matrix = find_nearest_positive_definite(eval_location_estimated_covariance_matrix)
        
        try:
            mvn_pdf = multivariate_normal.logpdf(
                x=y_test[test_data_point_index, :],
                mean=np.zeros(n_time_series),
                cov=eval_location_estimated_covariance_matrix,
                allow_singular=True  # TODO: this might be problematic
            )
        except ValueError:
            mvn_pdf = np.NaN  # TODO: does this introduce a bias?
        all_observation_log_likelihoods.append(mvn_pdf)
    all_observation_log_likelihoods = np.array(all_observation_log_likelihoods)  # (N_test, )
    mean_test_log_likelihood = np.average(
        all_observation_log_likelihoods[~np.isnan(all_observation_log_likelihoods)]
    )
    return mean_test_log_likelihood


def leave_every_other_out_split(full_array: np.array) -> (np.array, np.array):
    """
    Simple data split by leaving every other data point as either train or test set.
    """
    train_array = full_array[::2]
    test_array = full_array[1::2]
    return train_array, test_array


def interpolate_all_matrices(train_estimated_covs: np.array, n_test_time_steps: int) -> np.array:
    """
    Interpolate to estimate covariance matrices at test locations according to LEOO data split.

    :param train_estimated_covs: array of shape (N_train, D, D)
    :param n_test_time_steps:
    :return:
    """
    n_time_steps = train_estimated_covs.shape[0]
    n_time_series = train_estimated_covs.shape[1]
    test_estimated_covs = np.zeros((n_test_time_steps, n_time_series, n_time_series))
    for i in range(n_time_steps - 1):
        interpolated_matrix = _interpolate_matrices(
            train_estimated_covs[i, :, :],
            train_estimated_covs[i + 1, :, :]
        )
        test_estimated_covs[i, :, :] = interpolated_matrix
    test_estimated_covs[-1, :, :] = train_estimated_covs[-1, :, :]
    return test_estimated_covs


def _interpolate_matrices(matrix_i: np.array, matrix_j: np.array) -> np.array:
    return np.average((matrix_i, matrix_j), axis=0)


def get_tvfc_estimates(
    config_dict: dict,
    model_name: str,
    data_split: str,
    x_train: np.array,
    y_train: np.array,
    metric: str,
    scan_id: int = None,
    experiment_dimensionality: str = None,
    subject=None,
    noise_type: str = None,
    i_trial: int = None,
    covs_type: str = None,
) -> np.array:
    """
    Estimates train locations covariance structure.
    Note that there is no reason for WP to predict at the same locations, we could also predict at more locations.

    Parameters
    ----------
    config_dict: dict
    :return:
        estimated TVFC, array of shape (N, D, D).
    """
    n_time_series = y_train.shape[1]

    data_set_name = config_dict['data-set-name']
    match data_set_name:
        case 'd2' | 'd3d' | 'd3s' | 'd4s' | 'd6s' | 'd9s' | 'd15s' | 'd50s':
            assert noise_type is not None
            assert i_trial is not None
            assert covs_type is not None

            repetition_time = None

            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type, data_split,
                f'trial_{i_trial:03d}', model_name
            )
            wp_model_filename = f'{covs_type:s}.json'

            tvfc_estimates_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type, f'trial_{i_trial:03d}',
                'TVFC_estimates', data_split, metric, model_name
            )
            tvfc_estimates_filepath = os.path.join(
                tvfc_estimates_savedir, f'{covs_type:s}.csv'
            )

            # Fix renaming issue.
            if model_name in ['SVWP', 'VWP', 'SVWP_joint', 'VWP_joint']:
                if not os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                    logging.warning(f"WP model file {os.path.join(wp_model_savedir, wp_model_filename):s} not found.")
                    if covs_type == 'boxcar':
                        wp_model_filename = 'checkerboard.json'
            if not os.path.exists(tvfc_estimates_filepath):
                if covs_type == 'boxcar':
                    tvfc_estimates_filepath = os.path.join(
                        tvfc_estimates_savedir, 'checkerboard.csv'
                    )

        case 'HCP_PTN1200_recon2':
            assert experiment_dimensionality is not None
            assert scan_id is not None
            assert subject is not None

            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], 'saved_models', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, model_name
            )
            wp_model_filename = f'{subject:d}.json'
            repetition_time = config_dict['repetition-time']
            # tvfc_estimates_filepath = os.path.join(
            #     config_dict['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
            #     data_split, experiment_dimensionality, connectivity_metric, model_name, f'{subject:s}.csv'
            # )
        case 'rockland':
            pp_pipeline = 'custom_fsl_pipeline'
            assert subject is not None
            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], pp_pipeline, 'saved_models', data_split, model_name
            )
            wp_model_filename = f"{subject.removesuffix('.csv'):s}.json"
            repetition_time = config_dict['repetition-time']
        case _:
            logging.error(f"Dataset '{data_set_name:s}' not recognized.")
            return

    match model_name:
        case 'VWP' | 'VWP_joint':
            if os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                k = gpflow.kernels.Matern52()
                m = VariationalWishartProcess(
                    x_train, y_train,
                    nu=n_time_series,
                    kernel=k
                )
                m.load_from_params_dict(
                    savedir=wp_model_savedir,
                    model_name=wp_model_filename,
                )
                if metric == 'correlation':
                    all_covs_means, _ = m.predict_corr(x_train)  # Tensor of shape (N, D, D), _
                else:
                    all_covs_means, _ = m.predict_cov(x_train)  # Tensor of shape (N, D, D), _
                estimated_tvfc = all_covs_means.numpy()  # (N, D, D)
                del m
            else:
                logging.warning(f"VWP model not found in '{wp_model_savedir:s}'.")
                return
        case 'SVWP' | 'SVWP_joint':
            if os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                k = gpflow.kernels.Matern52()
                m = SparseVariationalWishartProcess(
                    D=n_time_series,
                    Z=x_train[:config_dict['n-inducing-points']],
                    nu=n_time_series,
                    kernel=k,
                    verbose=False
                )
                m.load_from_params_dict(
                    savedir=wp_model_savedir,
                    model_name=wp_model_filename,
                )
                if metric == 'correlation':
                    all_covs_means, _ = m.predict_corr(x_train)  # Tensor of shape (N, D, D), _
                else:
                    all_covs_means, _ = m.predict_cov(x_train)  # Tensor of shape (N, D, D), _
                estimated_tvfc = all_covs_means.numpy()  # (N, D, D)
                del m
            else:
                logging.warning(f"SVWP model not found in '{wp_model_savedir:s}'.")
                return
        case 'DCC' | 'DCC_joint' | 'DCC_bivariate_loop' | 'GO' | 'GO_joint' | 'GO_bivariate_loop':
            if os.path.exists(tvfc_estimates_filepath):
                mgarch_df = pd.read_csv(
                    tvfc_estimates_filepath,
                    index_col=0,
                )  # (D*D, N_train)
                estimated_tvfc = to_3d_format(mgarch_df.values)  # (N_train, D, D)
                logging.info(f"Loaded {model_name:s} estimates '{tvfc_estimates_filepath:s}'.")
                if data_split == 'LEOO':
                    estimated_tvfc = interpolate_all_matrices(
                        train_estimated_covs=estimated_tvfc,
                        n_test_time_steps=len(x_train)
                    )  # (N_test, D, D)
            else:
                logging.warning(f"MGARCH model '{model_name:s}' not found in '{tvfc_estimates_filepath:s}'.")
                return
        case 'SW_cross_validated':
            if os.path.exists(tvfc_estimates_filepath):
                sw_cross_validated_df = pd.read_csv(tvfc_estimates_filepath, index_col=0)  # (D*D, N_train)
                estimated_tvfc = to_3d_format(sw_cross_validated_df.values)  # (N_train, D, D)
                logging.info(f"Loaded {model_name:s} estimates '{tvfc_estimates_filepath:s}'.")
                if data_split == 'LEOO':
                    estimated_tvfc = interpolate_all_matrices(
                        train_estimated_covs=estimated_tvfc,
                        n_test_time_steps=len(x_train)
                    )  # (N_test, D, D)
            else:
                logging.warning(f"SW-CV model '{model_name:s}' not found in '{tvfc_estimates_filepath:s}'.")
                return
        case 'SW_15' | 'SW_16' | 'SW_30' | 'SW_60' | 'SW_120':
            window_length = int(model_name.removeprefix("SW_"))  # in seconds, TODO: change when using LEOO?
            window_length = int(np.floor(window_length / config_dict['repetition-time']))  # TODO: check this
            if data_split == 'LEOO':
                window_length = int(window_length / 2)
            sw = SlidingWindows(
                x_train_locations=x_train,
                y_train_locations=y_train,
                repetition_time=repetition_time,
            )
            estimated_tvfc = sw.overlapping_windowed_cov_estimation(
                window_length=window_length,
                repetition_time=config_dict['repetition-time']  # TODO: does this work with simulations?
            )  # (N_train, D, D)
            if metric == 'correlation':
                estimated_tvfc = to_correlation_structure(estimated_tvfc)  # (N_train, D, D)
        case 'sFC':
            sw = SlidingWindows(
                x_train_locations=x_train,
                y_train_locations=y_train,
                repetition_time=repetition_time
            )
            estimated_tvfc = sw.estimate_static_functional_connectivity(connectivity_metric=metric)  # (N_train, D, D)
        case _:
            logging.error(f"Model name '{model_name:s}' not recognized.")
            return

    return estimated_tvfc


def get_test_location_estimated_covariance_structure(
    config_dict: dict,
    model_name: str,
    n_time_series: int,
    x_train_locations: np.array,
    x_test_locations: np.array,
    subject: str,
    data_split: str,
    y_train_locations: np.array = None,
    scan_id: int = None,
    experiment_dimensionality: str = None,
    noise_type: str = None,
    i_trial: int = None,
    covs_type: str = None,
    connectivity_metric: str = 'covariance',
) -> np.array:
    """
    Estimates test location covariance structure.
    For all methods except the Wishart process models, we load the pre-saved train locations predictions.

    Parameters
    ----------
    config_dict: dict
    model_name: str
    experiment_dimensionality: str
        'multivariate' or 'bivariate'.
    """
    data_set_name = config_dict['data-set-name']
    match data_set_name:
        case 'd2' | 'd3d' | 'd3s' | 'd4s' | 'd6s' | 'd9s' | 'd15s' | 'd50s':

            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type, data_split,
                f'trial_{i_trial:03d}', model_name
            )
            wp_model_filename = f'{covs_type:s}.json'

            tvfc_estimates_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type, f'trial_{i_trial:03d}', 'TVFC_estimates',
                data_split, connectivity_metric, model_name
            )
            tvfc_estimates_filepath = os.path.join(
                tvfc_estimates_savedir, f'{covs_type:s}.csv'
            )

            # Fix renaming issue.
            if model_name in ['SVWP', 'VWP', 'SVWP_joint', 'VWP_joint']:
                if not os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                    logging.warning(f"WP model file {os.path.join(wp_model_savedir, wp_model_filename):s} not found.")
                    if covs_type == 'boxcar':
                        wp_model_filename = 'checkerboard.json'
            if not os.path.exists(tvfc_estimates_filepath):
                if covs_type == 'boxcar':
                    tvfc_estimates_filepath = os.path.join(
                        tvfc_estimates_savedir, 'checkerboard.csv'
                    )

        case 'HCP_PTN1200_recon2':
            assert scan_id is not None
            assert experiment_dimensionality is not None

            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], 'saved_models', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, model_name
            )
            subject = subject.removesuffix('.txt')
            wp_model_filename = f'{subject:s}.json'

            tvfc_estimates_filepath = os.path.join(
                config_dict['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, connectivity_metric, model_name, f'{subject:s}.csv'
            )
        case 'rockland':
            pp_pipeline = 'custom_fsl_pipeline'
            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], pp_pipeline, 'saved_models', data_split, model_name
            )
            wp_model_filename = f"{subject.removesuffix('.csv'):s}.json"
            tvfc_estimates_filepath = os.path.join(
                config_dict['experiments-basedir'], pp_pipeline, 'TVFC_estimates',
                data_split, connectivity_metric, model_name, subject
            )
        case _:
            logging.error(f"Dataset '{data_set_name:s}' not recognized.")
            return

    match model_name:
        case 'VWP' | 'VWP_joint':  # we do not have to interpolate linearly here
            wp_model_filepath = os.path.join(wp_model_savedir, wp_model_filename)
            if not os.path.exists(wp_model_filepath):
                raise FileNotFoundError(f"Could not load WP model '{wp_model_filepath:s}'.")
            k = gpflow.kernels.Matern52()
            m = VariationalWishartProcess(
                x_observed=x_train_locations,
                y_observed=y_train_locations,
                nu=n_time_series,
                kernel=k
            )
            m.load_from_params_dict(wp_model_savedir, wp_model_filename)
            if connectivity_metric == 'correlation':
                all_covs_means, _ = m.predict_corr(x_test_locations)  # Tensor of shape (N, D, D), _
            else:
                all_covs_means, _ = m.predict_cov(x_test_locations)  # Tensor of shape (N, D, D), _
            test_locations_predicted_covariance_structure = all_covs_means.numpy()  # (N, D, D)
            del m
        case 'SVWP' | 'SVWP_joint':  # we do not have to interpolate linearly here
            wp_model_filepath = os.path.join(wp_model_savedir, wp_model_filename)
            if not os.path.exists(wp_model_filepath):
                raise FileNotFoundError(f"Could not load WP model '{wp_model_filepath:s}'.")
            k = gpflow.kernels.Matern52()
            m = SparseVariationalWishartProcess(
                D=n_time_series,
                Z=np.linspace(x_train_locations[0], x_train_locations[-1], config_dict['n-inducing-points']),  # these will be overwritten by saved model
                nu=n_time_series,
                kernel=k,
                verbose=False
            )
            m.load_from_params_dict(
                savedir=wp_model_savedir,
                model_name=wp_model_filename,
            )
            test_locations_predicted_covariance_structure, _ = m.predict_cov(x_new=x_test_locations)  # Tensor of shape (N_test, D, D)
            test_locations_predicted_covariance_structure = test_locations_predicted_covariance_structure.numpy()  # (N_test, D, D)
        case _:
            if not os.path.exists(tvfc_estimates_filepath):
                logging.warning(f"Could not load TVFC estimates '{tvfc_estimates_filepath:s}'.")
                return
            # TODO: the Rockland load did not have an index_col=0 before!
            tvfc_estimates_df = pd.read_csv(
                tvfc_estimates_filepath,
                index_col=0,
            )  # (D*D, N_train)
            train_locations_predicted_covariance_structure = to_3d_format(tvfc_estimates_df.values)  # (N_train, D, D)
            test_locations_predicted_covariance_structure = interpolate_all_matrices(
                train_estimated_covs=train_locations_predicted_covariance_structure,
                n_test_time_steps=len(x_test_locations),
            )  # (N_test, D, D)
    return test_locations_predicted_covariance_structure
