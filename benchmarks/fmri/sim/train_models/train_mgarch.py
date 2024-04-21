import logging
import os
import socket
import sys

from fcest.models.mgarch import MGARCH
import pandas as pd

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    data_set_name = sys.argv[1]    # 'd2', 'd3d', 'd{%d}s'
    data_split = sys.argv[2]       # 'all', 'LEOO'
    experiment_data = sys.argv[3]  # e.g. 'N0200_T0001'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    n_trials = int(experiment_data[-4:])

    # Allow for local and CPU cluster training.
    # When running on the Hivemind with SLURM, only one model is trained here.
    if hostname == 'hivemind':
        try:
            i_trial = os.environ['SLURM_ARRAY_TASK_ID']
            # i_trial = os.environ['SLURM_ARRAY_JOB_ID']
            i_trial = int(i_trial) - 1  # to make zero-index
            print('SLURM trial ID', i_trial)

            assert len(sys.argv) == 6
            noise_type = sys.argv[4]
            covs_type = sys.argv[5]

            i_trials = [i_trial]
            noise_types = [noise_type]
            covs_types = [covs_type]
        except KeyError:
            i_trials = range(n_trials)
            noise_types = cfg['noise-types']
            covs_types = cfg['all-covs-types']
    else:
        print('Running locally...')
        i_trials = range(n_trials)
        noise_types = cfg['noise-types']
        covs_types = cfg['all-covs-types']

    for noise_type in noise_types:
        for covs_type in covs_types:
            for i_trial in i_trials:
                print('\n----------')
                print(f'Trial      {i_trial:d}')
                print('covs_type ', covs_type)
                print('noise_type', noise_type, '\n----------\n')
                data_file = os.path.join(
                    cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                    f'{covs_type:s}_covariance.csv'
                )
                if not os.path.exists(data_file):
                    if covs_type == 'boxcar':
                        data_file = os.path.join(
                            cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                            'checkerboard_covariance.csv'
                        )
                x, y = load_data(
                    data_file,
                    verbose=False
                )  # (N, 1), (N, D)
                n_time_series = y.shape[1]

                match data_split:
                    case "LEOO":
                        x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                        y_train, _ = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
                    case "all":
                        x_train = x  # (N, 1)
                        y_train = y  # (N, D)
                    case _:
                        raise NotImplementedError("Data split not recognized.")

                for model_name in cfg['mgarch-models']:

                    for training_type in cfg['mgarch-training-types']:

                        for metric in ['correlation', 'covariance']:

                            tvfc_estimates_savedir = os.path.join(
                                cfg['experiments-basedir'], noise_type, f'trial_{i_trial:03d}', 'TVFC_estimates',
                                data_split, metric, f'{model_name:s}_{training_type:s}'
                            )
                            tvfc_estimates_savepath = os.path.join(tvfc_estimates_savedir, f"{covs_type:s}.csv")
                            if not os.path.exists(tvfc_estimates_savepath):
                                m = MGARCH(mgarch_type=model_name)
                                m.fit_model(training_data_df=pd.DataFrame(y_train), training_type=training_type)
                                m.save_tvfc_estimates(
                                    savedir=tvfc_estimates_savedir,
                                    model_name=f'{covs_type:s}.csv',
                                    connectivity_metric=metric
                                )
                            else:
                                logging.info(f"Skipping training: existing model found in '{tvfc_estimates_savedir:s}'.")
