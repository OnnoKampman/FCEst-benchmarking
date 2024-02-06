import os
import sys
import socket

import pandas as pd

from configs.configs import get_config_dict
from helpers.kernel_analysis import plot_lengthscale_window_length_relation


if __name__ == "__main__":

    model_name = sys.argv[1]  # 'VWP_joint' or 'SVWP_joint'

    data_split = 'all'
    kernel_param = 'kernel_lengthscales'
    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )

    figures_savedir = os.path.join(cfg['figures-basedir'], pp_pipeline)
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis', data_split, model_name)
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

    # Remove outliers. 
    # TODO: must be a better way of doing this
    kernel_params_array[kernel_params_array > 0.4] = 0.4

    plot_lengthscale_window_length_relation(
        cfg, kernel_params_array, optimal_window_lengths_array,
        figures_savedir=figures_savedir
    )
