import logging
import os

from fcest.helpers.array_operations import get_all_lower_triangular_indices_tuples
import numpy as np
import pandas as pd

from helpers.array_operations import reconstruct_symmetric_summary_measure_matrix_from_tril
from helpers.icc import compute_icc_scores_pingouin
from helpers.i2c2 import compute_i2c2, to_i2c2_format


def compute_tvfc_summary_measure_test_retest_scores(
        config_dict: dict, test_retest_metric: str, model_name: str, tvfc_summary_measure: str,
        n_time_series: int, experiment_dimensionality: str, data_split: str, connectivity_metric: str
) -> pd.DataFrame:
    """
    Compute test-retest metrics.

    :param config_dict:
    :param test_retest_metric: 'ICC' or 'I2C2'
    :param model_name: 'SVWP_joint', 'DCC_joint', 'GO_joint', 'SW_cross_validated', 'SW_30', 'SW_60', 'sFC'
    :param summary_measure:
    :param n_time_series:
    :param experiment_dimensionality:
    :param data_split:
    :param connectivity_metric:
    :return:
    """
    n_scans = len(config_dict['scan-ids'])

    tvfc_summary_measure_array = np.zeros(
        (config_dict['n-subjects'], n_scans, n_time_series, n_time_series)
    )  # (n_subjects, n_scans, D, D)
    for i_scan_id, scan_id in enumerate(config_dict['scan-ids']):
        print(f"> TVFC '{tvfc_summary_measure:s}' - SCAN {i_scan_id+1:d} / {n_scans:d}")

        # We load the estimates directly, they are saved with another script.
        tvfc_estimates_summaries_savedir = os.path.join(
            config_dict['experiments-basedir'], 'TVFC_estimates_summary_measures', f'scan_{scan_id:d}',
            data_split, experiment_dimensionality, connectivity_metric, model_name
        )
        tvfc_estimates_summaries_df = pd.read_csv(
            os.path.join(tvfc_estimates_summaries_savedir, f"TVFC_{tvfc_summary_measure:s}.csv"),
            index_col=0
        )  # (n_subjects, D*(D-1)/2)

        print(tvfc_estimates_summaries_df)  # may contain empty rows!
        if tvfc_estimates_summaries_df.isnull().any(axis=1).any():
            logging.warning("No TVFC estimate summary measures found for some subject(s).")

        tvfc_estimates_summaries = [
            reconstruct_symmetric_summary_measure_matrix_from_tril(
                tvfc_summaries_vector,
                tvfc_summary_measure=tvfc_summary_measure,
                n_time_series=n_time_series
            ) for tvfc_summaries_vector in tvfc_estimates_summaries_df.values
        ]  # (n_subjects, D, D)

        tvfc_summary_measure_array[:, i_scan_id, :, :] = tvfc_estimates_summaries

    if test_retest_metric == 'ICC':
        icc_edgewise_array = _get_icc_array(tvfc_summary_measure_array)  # (D, D)
        return pd.DataFrame(icc_edgewise_array)
    elif test_retest_metric == 'I2C2':
        tvfc_summary_measure_array = to_i2c2_format(tvfc_summary_measure_array)  # (n_subject*n_scans, D*(D-1)/2)
        i2c2_score = compute_i2c2(
            y=tvfc_summary_measure_array,
            n_subjects=config_dict['n-subjects'],
            n_scans=n_scans
        )
        return i2c2_score
    else:
        logging.error(f"Test-retest metric '{test_retest_metric:s}' not recognized.")


def _get_icc_array(tvfc_summary_measure_array: np.array) -> np.array:
    """
    TODO: should we use ICC2k instead?

    :param tvfc_summary_measure_array: array of shape (n_subjects, n_scans, D, D).
    :return:
    """
    n_time_series = tvfc_summary_measure_array.shape[2]  # D
    icc_array = np.zeros((n_time_series, n_time_series))
    for (i, j) in get_all_lower_triangular_indices_tuples(n_time_series):
        cov_structure_summary_measure = tvfc_summary_measure_array[:, :, i, j]  # (n_subjects, n_scans)

        icc_score = compute_icc_scores_pingouin(cov_structure_summary_measure, icc_type='ICC2')
        # icc_score = ICC_rep_anova(cov_structure_summary_measure)[0]

        icc_array[i, j] = icc_score
    return icc_array
