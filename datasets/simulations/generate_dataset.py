import logging
import os
import socket
import sys

import numpy as np

from configs.configs import get_config_dict
from helpers.simulations import save_synthetic_dataset
from helpers.synthetic_covariance_structures import get_constant_covariances
from helpers.synthetic_covariance_structures import get_periodic_covariances
from helpers.synthetic_covariance_structures import get_stepwise_covariances
from helpers.synthetic_covariance_structures import get_state_transition_covariances
from helpers.synthetic_covariance_structures import get_boxcar_covariances
from helpers.synthetic_covariance_structures import get_change_point_covariances
from helpers.synthetic_covariance_structures import get_d2_covariance_structure
from helpers.synthetic_covariance_structures import get_d3d_covariance_structure, get_sparse_covariance_structure

np.random.seed(2021)  # so full data set can be replicated


def _define_noise_type_name(white_noise_snr: float, hcp_noise_snr: float) -> str:
    if white_noise_snr is None and hcp_noise_snr is None:
        noise_type_name = 'no_noise'
    elif white_noise_snr is not None and hcp_noise_snr is not None:
        logging.error('Cannot add both white noise and HCP noise.')
    elif white_noise_snr is not None and hcp_noise_snr is None:
        noise_type_name = f'white_noise_snr_{white_noise_snr:d}'
    elif white_noise_snr is None and hcp_noise_snr is not None:
        noise_type_name = f'HCP_noise_snr_{hcp_noise_snr:d}'
    else:
        noise_type_name = None
    return noise_type_name


if __name__ == "__main__":

    # sys.argv[0] is the script name
    data_set_name = sys.argv[1]  # 'd2', 'd3d', or 'd{%d}s'
    N = int(sys.argv[2])         # number of time steps
    n_trials = int(sys.argv[3])  # number of trials

    print(f'N = {N:d}')
    print(f'T = {n_trials:d}')
    experiment_data = f'N{N:04d}_T{n_trials:04d}'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    if os.path.exists(cfg['data-dir']):
        raise FileExistsError('Data set already exists!')

    match [data_set_name[0], data_set_name[1:-1], data_set_name[-1]]:
        case ['d', '2', 's' | 'd'] | ['d', '', '2']:  # 'd2', 'd2d', or 'd2s'
            get_covariance_structure = get_d2_covariance_structure
            n_time_series = 2
        case ['d', '3', 'd']:  # 'd3d'
            get_covariance_structure = get_d3d_covariance_structure
            n_time_series = 3
        case ['d', n_time_series, 's']:  # seq of 3 elems: 'd', anything, 's', e.g. 'd3s'
            get_covariance_structure = get_sparse_covariance_structure
            n_time_series = int(n_time_series)
        case _:
            raise NotImplementedError(f"Data set name '{data_set_name:s}' not recognized.")

    for (white_noise_snr, hcp_noise_snr) in cfg['noise-routines']:
        noise_type = _define_noise_type_name(white_noise_snr, hcp_noise_snr)
        for i_trial in range(n_trials):
            data_dir = os.path.join(cfg['data-dir'], noise_type, f'trial_{i_trial:03d}')

            # Null data set.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_constant_covariances(n_samples=N, covariance=0),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='null_covariance.csv',
                synthetic_data_dir=data_dir
            )

            # Constant data set.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_constant_covariances(n_samples=N, covariance=cfg['constant-covariance']),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='constant_covariance.csv',
                synthetic_data_dir=data_dir
            )

            # Periodic data sets.
            for n_periods in [1, 2, 3, 4, 5]:
                save_synthetic_dataset(
                    config_dict=cfg,
                    covariance_structure=get_covariance_structure(
                        get_periodic_covariances(n_samples=N, n_periods=n_periods),
                        n_time_series=n_time_series
                    ),
                    white_noise_snr=white_noise_snr,
                    hcp_noise_snr=hcp_noise_snr,
                    dataset_name=f'periodic_{n_periods:d}_covariance.csv',
                    synthetic_data_dir=data_dir
                )

            # Stepwise covariance.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_stepwise_covariances(n_samples=N),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='stepwise_covariance.csv',
                synthetic_data_dir=data_dir
            )

            # State transition covariance.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_state_transition_covariances(n_samples=N),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='state_transition_covariance.csv',
                synthetic_data_dir=data_dir
            )

            # Boxcar covariance.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_boxcar_covariances(n_samples=N),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='boxcar_covariance.csv',
                synthetic_data_dir=data_dir
            )

            # Change point covariance.
            save_synthetic_dataset(
                config_dict=cfg,
                covariance_structure=get_covariance_structure(
                    get_change_point_covariances(n_samples=N),
                    n_time_series=n_time_series
                ),
                white_noise_snr=white_noise_snr,
                hcp_noise_snr=hcp_noise_snr,
                dataset_name='change_point_covariance.csv',
                synthetic_data_dir=data_dir
            )
