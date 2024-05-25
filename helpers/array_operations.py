import logging

import numpy as np


def get_n_lower_triangular_values(n_time_series: int) -> int:
    return int(n_time_series * (n_time_series - 1) / 2)


def slice_covariance_structure(full_covariance_structure: np.array, edge_indices: tuple) -> np.array:
    """
    Select a certain edge from a full covariance structure.

    :param full_covariance_structure:
    :param edge_indices:
    :return:
    """
    sliced_covariance_structure = full_covariance_structure[:, edge_indices][:, :, edge_indices]
    return sliced_covariance_structure


def map_array(original_array: np.array, array_map: dict) -> np.array:
    mapped_array = np.array([array_map[arr] for arr in original_array])
    return mapped_array


def reconstruct_symmetric_matrix_from_tril(
        cluster_array: np.array, n_time_series: int,
        diagonal: str = 'ones'
) -> np.array:
    """
    Reconstruct full matrix from lower triangular values.
    TODO: can we use scipy.spatial.distance.squareform instead? like m = squareform(upper_vector)? diagonals will be zeros in that case

    :param cluster_array:
    :param n_time_series:
    :param diagonal:
    :return: array of shape (D, D)
    """
    match diagonal:
        case 'ones':
            reconstructed_corr_matrix = np.ones(shape=(n_time_series, n_time_series))  # (D, D)
        case 'zeros':
            reconstructed_corr_matrix = np.zeros(shape=(n_time_series, n_time_series))  # (D, D)
        case _:
            reconstructed_corr_matrix = np.ones(shape=(n_time_series, n_time_series))  # (D, D)

    mask = np.tri(n_time_series, dtype=bool, k=-1)  # matrix of bools

    # Add lower triangular values.
    reconstructed_corr_matrix[mask] = cluster_array

    # Add upper triangular values (transpose matrix first and then re-add lower triangular values).
    reconstructed_corr_matrix = reconstructed_corr_matrix.T
    reconstructed_corr_matrix[mask] = cluster_array

    assert _check_symmetric(reconstructed_corr_matrix)

    return reconstructed_corr_matrix


def reconstruct_symmetric_summary_measure_matrix_from_tril(
    cluster_array: np.array,
    num_time_series: int,
    tvfc_summary_measure: str,
) -> np.array:
    """
    This propagates NaN values if present.
    TODO: perhaps we could merge this with nilearn.connectome.vec_to_sym_matrix

    Parameters
    ----------
    :param cluster_array:
    :param num_time_series:
        Denoted as D.
    :param tvfc_summary_measure:
    :return:
        Array of shape (D, D).
    """
    if np.isnan(cluster_array).any():
        return np.full(shape=[num_time_series, num_time_series], fill_value=np.nan)
    match tvfc_summary_measure:
        case 'ar1':
            return reconstruct_symmetric_matrix_from_tril(cluster_array, num_time_series, diagonal='zeros')
        case 'mean':
            return reconstruct_symmetric_matrix_from_tril(cluster_array, num_time_series, diagonal='ones')
        case 'rate_of_change':
            return reconstruct_symmetric_matrix_from_tril(cluster_array, num_time_series, diagonal='zeros')
        case 'std':
            return reconstruct_symmetric_matrix_from_tril(cluster_array, num_time_series, diagonal='zeros')
        case 'variance':
            return reconstruct_symmetric_matrix_from_tril(cluster_array, num_time_series, diagonal='zeros')
        case _:
            logging.error(f"Summary measure '{tvfc_summary_measure:s}' not recognized.")


def reorder_symmetric_matrix(original_matrix: np.array, new_order: list) -> np.array:
    assert _check_symmetric(original_matrix)
    reordered_matrix = original_matrix[:, new_order][new_order]
    assert _check_symmetric(reordered_matrix)
    return reordered_matrix


def reorder_lower_triangular_matrix(original_matrix: np.array, new_order: list) -> np.array:
    """
    The upper triangular values could still be set to zeros if needed.

    :param original_matrix:
    :param new_order:
    :return:
    """
    fully_filled_matrix = original_matrix + original_matrix.T
    return fully_filled_matrix[:, new_order][new_order]


def _check_symmetric(a: np.array, rtol=1e-05, atol=1e-08) -> bool:
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def convert_to_seconds(
    normalized_array: np.array, repetition_time: float, data_length: int
) -> np.array:
    """
    :param normalized_array:
    :param repetition_time: in seconds.
    :param data_length:
    :return:
    """
    return normalized_array * repetition_time * data_length


def convert_to_minutes(
    normalized_array: np.array, repetition_time: float, data_length: int
) -> np.array:
    """
    :param normalized_array:
    :param repetition_time: in seconds.
    :param data_length:
    :return:
    """
    return convert_to_seconds(
        normalized_array, repetition_time=repetition_time, data_length=data_length
    ) / 60
