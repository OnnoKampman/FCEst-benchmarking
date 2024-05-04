import os
import sys
import socket

import pandas as pd

from configs.configs import get_config_dict
from helpers.kernel_analysis import plot_lengthscale_window_length_relation


if __name__ == "__main__":

    data_split = 'all'
    kernel_param = 'kernel_lengthscales'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_time_series = int(data_dimensionality[1:])
    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis')
    kernel_params_df = pd.read_csv(
        os.path.join(kernel_params_savedir, f'{kernel_param:s}_kernel_params.csv'),
        index_col=0
    )
    print(kernel_params_df)
    optimal_window_lengths_df = pd.read_csv(
        os.path.join(cfg['git-results-basedir'], 'optimal_window_lengths', data_split, 'optimal_window_lengths.csv'),
        index_col=0
    )
    print(optimal_window_lengths_df)

    # Prepare data for plot.
    assert kernel_params_df.shape == optimal_window_lengths_df.shape
    kernel_params_array = kernel_params_df.values.reshape(-1, 1)
    optimal_window_lengths_array = optimal_window_lengths_df.values.reshape(-1, 1)

    plot_lengthscale_window_length_relation(
        cfg, kernel_params_array, optimal_window_lengths_array,
        figures_savedir=cfg['figures-basedir']
    )
