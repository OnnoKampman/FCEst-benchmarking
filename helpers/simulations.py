import logging
import os

import numpy as np
import pandas as pd

from helpers.data import normalize_array, assert_normalized


def save_synthetic_dataset(
        config_dict: dict, covariance_structure: np.array, dataset_name: str,
        synthetic_data_dir: str, white_noise_snr=None, hcp_noise_snr=None
) -> None:
    """
    Saves synthetic data time series to `.csv` format.
    This is preferred as our R scripts need to be trained on the same data as the Python-based methods.
    We do not save the covariance structures to any file, since these are deterministic and can be re-created easily.

    :param config_dict:
    :param covariance_structure: of shape (N, D, D).
    :param dataset_name:
    :param synthetic_data_dir:
    :param white_noise_snr:
    :param hcp_noise_snr:
    :return:
    """
    y = simulate_time_series(covariance_structure=covariance_structure)  # (N, D)
    n_time_steps = y.shape[0]
    n_time_series = y.shape[1]

    # Randomly select HCP noise.
    if hcp_noise_snr is not None:
        all_hcp_noise = _pick_random_hcp_noise(
            config_dict=config_dict,
            n_samples=n_time_steps,
            n_time_series=n_time_series
        )  # (N, D)

    # Normalize data and add noise.
    for i_timeseries in range(n_time_series):
        y[:, i_timeseries] = normalize_array(y[:, i_timeseries])

        # Add white noise.
        if white_noise_snr is not None:
            white_noise_ts = _generate_white_noise_ts(n_samples=n_time_steps)  # (N, )
            y[:, i_timeseries] = _mix_signals_linearly(
                y[:, i_timeseries], white_noise_ts, white_noise_snr
            )

        # Add HCP noise.
        if hcp_noise_snr is not None:
            hcp_noise_ts = all_hcp_noise[:, i_timeseries]  # (N, )
            y[:, i_timeseries] = _mix_signals_linearly(
                y[:, i_timeseries], hcp_noise_ts, hcp_noise_snr
            )

        assert_normalized(y[:, i_timeseries])

    # Save time series to disk.
    df = pd.DataFrame(
        y,
        columns=[f"ts_{i:02d}" for i in range(n_time_series)]
    )
    if not os.path.exists(synthetic_data_dir):
        os.makedirs(synthetic_data_dir)
    df.to_csv(
        os.path.join(synthetic_data_dir, dataset_name),
        index=False, header=True
    )
    logging.info(f"Saved synthetic data '{dataset_name:s}' in '{synthetic_data_dir:s}'.")


def simulate_time_series(covariance_structure: np.array) -> np.array:
    """
    Generates random time series data based on a given (synthetic) covariance structure.

    :param covariance_structure: full covariance structure of shape (N, D, D).
    :return
        y: array of shape (n_time_steps, n_time_series) or (N, D).
    """
    n_time_steps = covariance_structure.shape[0]
    n_time_series = covariance_structure.shape[1]

    means = np.zeros(n_time_series)
    y = []
    for sample_step in range(n_time_steps):
        time_step_covariance_matrix = covariance_structure[sample_step, :, :]
        try:
            _ = np.linalg.cholesky(time_step_covariance_matrix)  # A = L*L^T
        except np.linalg.LinAlgError:
            logging.error('Matrix not positive definite!')
            print(time_step_covariance_matrix)
        time_step_observation = np.random.multivariate_normal(
            mean=means,
            cov=time_step_covariance_matrix,
            size=1
        ).astype(np.float64).flatten()
        y.append(time_step_observation)

    return np.array(y)


def _pick_random_hcp_noise(
        config_dict: dict, n_samples: int, n_time_series: int,
        total_samples: int = 1200
) -> np.array:
    """
    We pick D unique subjects from all available subjects.
    Then we pick a random node time series for each subject.
    We select the middle N samples from the first resting state scan.
    We normalize this selection to N(0, 1).

    :param config_dict:
    :param n_samples:
    :param n_time_series: D
    :param total_samples: the number of time steps in the HCP data, which is 1200.
        We take samples from middle, e.g. take 200 samples starting at an offset of 500.
    :return: noise time series normalized to mean 0 and stddev 1.
    """
    hcp_time_series_dir = os.path.join(
        config_dict['hcp-data-dir'], 'HCP_PTN1200_recon2', 'node_timeseries', '3T_HCP1200_MSMAll_d15_ts2'
    )
    start_offset = int((total_samples - n_samples) / 2)

    # Select unique subjects.
    all_subjects_list = os.listdir(hcp_time_series_dir)  # list of all files in this directory
    random_subjects = np.random.choice(all_subjects_list, n_time_series, replace=False)

    picked_noise_time_series = []
    for i_subject, subject in enumerate(random_subjects):
        df = pd.read_csv(
            os.path.join(hcp_time_series_dir, subject),
            delimiter=' ', header=None
        )
        time_series_index = np.random.choice(df.shape[1], 1)[0]  # pick random time series
        subject_time_series = df.values[start_offset:(start_offset + n_samples), time_series_index]

        # We cannot do an analytic normalization here.
        subject_time_series = normalize_array(subject_time_series)

        picked_noise_time_series.append(subject_time_series)
    return np.array(picked_noise_time_series).T  # (N, D)


def _generate_white_noise_ts(n_samples: int, mean=0, std=1) -> np.array:
    return np.random.normal(mean, std, n_samples)


def _mix_signals_linearly(
        original_time_series: np.array, added_time_series: np.array, added_ts_snr
) -> np.array:
    """
    We could normalize the signal in several ways here.
    TODO: which normalization form should we use?

    :param original_time_series:
    :param added_time_series:
    :param added_ts_snr:
    :return:
    """
    mixing_coefficient = added_ts_snr / (added_ts_snr + 1)  # \alpha
    mixed_time_series = mixing_coefficient * original_time_series + (1 - mixing_coefficient) * added_time_series

    mixed_time_series = normalize_array(mixed_time_series)
    # mixed_time_series = _normalize_array_analytic(mixed_time_series, mixing_coefficient=mixing_coefficient)

    return mixed_time_series
