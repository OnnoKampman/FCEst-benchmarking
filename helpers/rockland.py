import logging
import os

import numpy as np
import pandas as pd

from helpers.data import normalize_array


def load_rockland_data(data_file: str, verbose: bool = True) -> (np.array, np.array):
    """
    Load Rockland BOLD time series from .csv file.

    :param data_file:
    :param verbose:
    :return:
    """
    df = pd.read_csv(data_file)
    logging.info(f"Loaded data from '{data_file:s}'.")

    if verbose:
        print(df.head())
        print(df.shape)

    n_time_steps = len(df)
    xx = np.linspace(0, 1, n_time_steps).reshape(-1, 1).astype(np.float64)

    return xx, df.values


def get_convolved_stim_array(
    config_dict: dict, pp_pipeline: str = 'custom_fsl_pipeline'
) -> np.array:
    convolved_stim_datadir = os.path.join(
        config_dict['data-basedir'], pp_pipeline, 'results', config_dict['roi-list-name']
    )
    convolved_stimulus_df = pd.read_csv(
        os.path.join(convolved_stim_datadir, 'convolved_stim.csv'),
        index_col=None,
        header=None
    )
    return convolved_stimulus_df.values.flatten()


def get_stimulus_array(
    config_dict: dict, pp_pipeline: str = 'custom_fsl_pipeline'
) -> np.array:
    stimulus_datadir = os.path.join(
        config_dict['data-basedir'], pp_pipeline, 'results', config_dict['roi-list-name']
    )
    stimulus_df = pd.read_csv(
        os.path.join(stimulus_datadir, 'stim.csv'),
        index_col=None,
        header=None
    )
    return stimulus_df.values.flatten()


def get_rockland_subjects(
        config_dict: dict,
        cutoff_v1_stim_correlation: float = None, first_n_subjects: int = None
) -> list:
    """
    Returns the full list of Rockland subjects.
    TODO: add option to leave out subjects that seem asleep or whose V1 time series do not correlate well with the HRF of the stimulus

    :param config_dict:
    :param cutoff_v1_stim_correlation: leave out subjects with a V1-stimulus correlation below this threshold. These
        subjects likely had their eyes closed, or the data was corrupted in another way.
    :param first_n_subjects: can be used to reduce the computational burden
    :return:
    """
    if cutoff_v1_stim_correlation is None:
        data_dir = os.path.join(
            config_dict['data-basedir'], 'custom_fsl_pipeline', 'node_timeseries', config_dict['roi-list-name']
        )
        # Get list of all files in this directory, alphabetically sorted.
        all_subject_filenames_list = sorted(os.listdir(data_dir))
    else:
        # Load correlations results.
        correlations_filename = "V1_stimulus_correlations.csv"
        correlation_results_df = pd.read_csv(
            os.path.join(config_dict["git-results-basedir"], correlations_filename),
            index_col=0
        )
        # Get list of sufficiently correlated subjects.
        correlation_results_df = correlation_results_df[correlation_results_df['V1-stim_correlation'] > cutoff_v1_stim_correlation]
        print(correlation_results_df)
        all_subject_filenames_list = correlation_results_df.index.values
    if first_n_subjects is not None:
        all_subject_filenames_list = all_subject_filenames_list[:first_n_subjects]
    logging.info(f"Found {len(all_subject_filenames_list):d} subjects in total.")
    return all_subject_filenames_list


def extract_time_series_rockland(subject_df: pd.DataFrame, regions_of_interest) -> pd.DataFrame:
    """
    Extract DataFrame with time series.

    :param subject_df:
    :param regions_of_interest:
    :return:
    """
    first_ts = subject_df[subject_df['results2'] == '{}_ts'.format(regions_of_interest[0])]['ts'].values[0]
    second_ts = subject_df[subject_df['results2'] == '{}_ts'.format(regions_of_interest[1])]['ts'].values[0]
    data = np.array([
        first_ts,
        second_ts
    ]).T  # (N, D)
    for i_ts in range(data.shape[1]):
        data[:, i_ts] = normalize_array(data[:, i_ts])
    bivariate_df = pd.DataFrame(
        data,
        columns=regions_of_interest
    )
    return bivariate_df


def get_edges_names(config_dict: dict) -> list:
    brain_regions_of_interest = config_dict['roi-list']
    edges_of_interest_indices = config_dict['roi-edges-list']
    edge_names = brain_regions_of_interest[edges_of_interest_indices]
    edge_names = ['-'.join(edge_names_pair) for edge_names_pair in edge_names]
    return edge_names
