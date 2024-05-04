import json
import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    data_set_name = sys.argv[1]    # 'd2', 'd3d', 'd3s'
    data_split = sys.argv[2]       # 'all', 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'
    model_name = sys.argv[4]       # 'SVWP', 'VWP', 'SVWP_joint', 'VWP_joint'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    num_trials = int(experiment_data[-4:])

    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis')
    if not os.path.exists(kernel_params_savedir):
        os.makedirs(kernel_params_savedir)

    i_trials = range(num_trials)
    noise_types = cfg['noise-types']
    covs_types = cfg['all-covs-types']

    for kernel_param in cfg['kernel-params']:

        for noise_type in noise_types:

            kernel_params_df = pd.DataFrame()  # (num_trials, num_covs_types)
            for covs_type in covs_types:

                kernel_params_array = []
                for i_trial in i_trials:
                    model_savedir = os.path.join(
                        cfg['experiments-basedir'], noise_type, data_split, f'trial_{i_trial:03d}', model_name
                    )
                    model_savepath = os.path.join(model_savedir, f'{covs_type:s}.json')
                    if os.path.exists(model_savepath):
                        with open(model_savepath) as f:
                            m_dict = json.load(f)
                    else:
                        logging.warning(f"Model '{model_savepath:s}' not found.")
                    kernel_params_df.loc[i_trial, covs_type] = m_dict[kernel_param]
            kernel_params_df_filename = f'{model_name:s}_{kernel_param:s}_kernel_params.csv'
            kernel_params_savedir = os.path.join(cfg['git-results-basedir'], noise_type, data_split)
            if not os.path.exists(kernel_params_savedir):
                os.makedirs(kernel_params_savedir)
            kernel_params_df.to_csv(
                os.path.join(kernel_params_savedir, kernel_params_df_filename),
                float_format='%.3f'
            )
            logging.info(f"Saved kernel params table '{kernel_params_df_filename:s}' to '{kernel_params_savedir:s}'.")
