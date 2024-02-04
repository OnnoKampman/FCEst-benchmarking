import logging
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from statsmodels.stats.correlation_tools import cov_nearest


def get_tvfc_estimates_similarity_matrix(
        config_dict: dict, tvfc_summary_measure: str, model_name: str,
        data_split: str = 'all', experiment_dimensionality: str = 'multivariate', connectivity_metric: str = 'correlation'
) -> np.array:
    """
    Computes a subject similarity matrix based on how 'similar' their estimated TVFC covariance structure is,
    based on three summary measures of it.
    TODO: should we take the average across scans before or after computing the similarity matrix?
    Only applicable to HCP data at the moment.

    :param config_dict:
    :param tvfc_summary_measure: 'mean', 'variance', 'std', or 'rate_of_change'.
    :param model_name: e.g. SVWP_joint
    :param data_split:
    :param experiment_dimensionality:
    :param connectivity_metric: 'correlation' or 'covariance'
    :return:
        array of shape (n_subjects, n_subjects)
    """
    all_subjects_tvfc_summary_measure_array = []
    for scan_id in config_dict['scan-ids']:
        tvfc_estimates_summaries_filepath = os.path.join(
            config_dict['experiments-basedir'], 'TVFC_estimates_summary_measures', f'scan_{scan_id:d}',
            data_split, experiment_dimensionality, connectivity_metric, model_name, f'TVFC_{tvfc_summary_measure:s}.csv'
        )
        if not os.path.exists(tvfc_estimates_summaries_filepath):
            logging.warning(f"TVFC estimates '{tvfc_estimates_summaries_filepath:s}' not found.")
            return None
        tvfc_estimates_summaries_df = pd.read_csv(
            tvfc_estimates_summaries_filepath, index_col=0
        )  # (n_subjects, D*(D-1)/2)
        all_subjects_tvfc_summary_measure_array.append(tvfc_estimates_summaries_df.values)
    all_subjects_tvfc_summary_measure_array = np.array(all_subjects_tvfc_summary_measure_array)  # (n_scans, n_subjects, D*(D-1)/2)

    # Take the mean TVFC estimates over all scans.
    all_subjects_tvfc_summary_measure_array = np.mean(all_subjects_tvfc_summary_measure_array, axis=0)  # (n_subjects, D*(D-1)/2)

    subject_tvfc_distance_summary_measure = _compute_similarity_matrix(
        all_subjects_data=all_subjects_tvfc_summary_measure_array
    )
    return subject_tvfc_distance_summary_measure


def get_kernel_lengthscale_similarity_matrix(
        config_dict: dict
) -> np.array:
    """
    Computes a subject similarity matrix based on how 'similar' their learned kernel lengthscales are, based on 4 scans.
    TODO: should we take the average across scans before or after computing the similarity matrix?

    :param config_dict:
    :return:
        array of shape (n_subjects, n_subjects)
    """
    kernel_lengthscales_filepath = os.path.join(
        config_dict['git-results-basedir'], 'kernel_analysis', 'kernel_lengthscales_kernel_params.csv'
    )  # (n_subjects, n_scans)
    kernel_lengthscales_df = pd.read_csv(kernel_lengthscales_filepath, index_col=0)
    print(kernel_lengthscales_df.shape)

    kernel_lengthscales_df = kernel_lengthscales_df.mean(axis=1)  # (n_subjects, )
    kernel_lengthscales_df = kernel_lengthscales_df.values.reshape(-1, 1)  # (n_subjects, 1)

    subject_kernel_lengthscale_similarity_matrix = _compute_similarity_matrix(kernel_lengthscales_df)
    return subject_kernel_lengthscale_similarity_matrix


def get_brain_state_switch_count_similarity_matrix(
        config_dict: dict, n_basis_states: int, tvfc_estimation_method: str
) -> np.array:
    """
    Computes a subject similarity matrix based on how 'similar' the number of brain state switches are,
    based on 4 scans.
    TODO: should we take the average across scans before or after computing the similarity matrix?

    :param config_dict:
    :param n_basis_states:
    :param tvfc_estimation_method:
    :return:
        array of shape (n_subjects, n_subjects)
    """
    n_brain_state_switches_filepath = os.path.join(
        config_dict['git-results-basedir'], 'brain_states', f'k{n_basis_states:d}', f'number_of_brain_state_switches_{tvfc_estimation_method:s}.csv'
    )  # (n_subjects, n_scans)
    n_brain_state_switches_df = pd.read_csv(n_brain_state_switches_filepath, index_col=0)
    print(n_brain_state_switches_df.shape)

    n_brain_state_switches_df = n_brain_state_switches_df.mean(axis=1)  # (n_subjects, )
    n_brain_state_switches_df = n_brain_state_switches_df.values.reshape(-1, 1)  # (n_subjects, 1)

    return _compute_similarity_matrix(n_brain_state_switches_df)


def _compute_similarity_matrix(
        all_subjects_data: np.array, distance_measure: str = 'Gaussian'
) -> np.array:
    """
    Compute similarity matrix (K), where entries encode the distance (Gaussian kernel, not linear) between subjects.

    :param all_subjects_data: array of shape (n_subjects, n_dimensions)
    :param distance_measure: 'Gaussian' (default) or 'correlation'
        Li et al. (2019) used correlation between subject vectors
    :return:
    """
    # Compute pairwise distances.
    pairwise_dists = squareform(pdist(all_subjects_data, 'euclidean'))

    match distance_measure:
        case 'Gaussian':
            # Compute Gaussian distance similarity matrix.
            sigma = 1
            similarity_matrix = np.exp(-pairwise_dists**2 / sigma**2)  # (n_subjects, n_subjects)
        case 'correlation':
            # Compute alternative similarity matrix.
            if distance_measure == 'correlation':
                similarity_matrix = cov_nearest(1 - (pairwise_dists / np.max(pairwise_dists)))  # (n_samples, n_samples)

    return similarity_matrix
