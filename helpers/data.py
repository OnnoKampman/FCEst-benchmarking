import logging

import numpy as np
import pandas as pd

from helpers.array_operations import map_array, reorder_symmetric_matrix, reorder_lower_triangular_matrix


def load_data(data_file: str, verbose: bool = True) -> (np.array, np.array):
    """
    :param data_file:
    :param verbose:
    :return:
    """
    df = pd.read_csv(data_file)
    if verbose:
        logging.info(f"Loaded data from '{data_file:s}'.")
        print(df.head())
        print(df.shape)
    num_time_steps = len(df)
    xx = np.linspace(0, 1, num_time_steps).reshape(-1, 1).astype(np.float64)
    return xx, df.values


def reorder_ica_components(
    config_dict: dict,
    original_matrix: np.array,
    num_time_series: int,
    lower_triangular: bool = False,
):
    # Get ICA RSN labels.
    original_index = np.arange(num_time_series)
    original_rsn_id_assignment_array = map_array(
        original_index,
        array_map=config_dict['ica-id-to-rsn-id-algo-map']
    )
    original_rsn_names = map_array(
        original_rsn_id_assignment_array,
        array_map=config_dict['rsn-id-to-functional-region-map']
    )

    # Re-order correlation matrix.
    new_rsn_ordering = np.argsort(original_rsn_id_assignment_array)
    new_index = original_index[new_rsn_ordering]
    new_rsn_assignment_array = map_array(
        new_index,
        array_map=config_dict['ica-id-to-rsn-id-algo-map']
    )
    new_rsn_names = map_array(
        new_rsn_assignment_array,
        array_map=config_dict['rsn-id-to-functional-region-map']
    )
    if not lower_triangular:
        reordered_matrix = reorder_symmetric_matrix(
            original_matrix,
            new_order=new_index
        )  # (D, D)
    else:
        reordered_matrix = reorder_lower_triangular_matrix(
            original_matrix,
            new_order=new_index
        )  # (D, D)
    return reordered_matrix, new_rsn_names


def normalize_array(original_array: np.array, verbose: bool = True) -> np.array:
    """
    Normalizes an array to have mean zero and standard deviation one.

    :param original_array: array of shape (N, ).
    :return: array of shape (N, ).
    """
    normalized_array = original_array - original_array.mean()
    if normalized_array.std() != 0:
        normalized_array = normalized_array / normalized_array.std()
        assert_normalized(normalized_array)
    else:
        logging.warning("Found standard deviation of 0.")
        print(f"before normalization: min = {original_array.min():.2f}, max = {original_array.max():.2f}, mean = {original_array.mean():.2f}, std = {original_array.std():.2f}")
        print(f"after  normalization: min = {normalized_array.min():.2f}, max = {normalized_array.max():.2f}")
    return normalized_array


def _normalize_array_analytic(original_array: np.array, mixing_coefficient: float) -> np.array:

    print('before analytic normalization')
    print('min', original_array.min())
    print('max', original_array.max())
    print('avg', original_array.mean())
    print('std', original_array.std())
    print('')

    normalized_array = original_array / np.sqrt(mixing_coefficient**2 + (1 - mixing_coefficient)**2)

    print('after analytic normalization')
    print('min', normalized_array.min())
    print('max', normalized_array.max())
    print('')

    assert_normalized(normalized_array)

    return normalized_array


def assert_normalized(time_series_array: np.array) -> None:
    """
    Make sure a time series is still normalized to zero-mean and a standard deviation of 1.

    :param time_series_array: array of shape (N, ).
    """
    np.testing.assert_almost_equal(np.mean(time_series_array), 0, decimal=2)
    np.testing.assert_almost_equal(np.std(time_series_array), 1, decimal=2)
