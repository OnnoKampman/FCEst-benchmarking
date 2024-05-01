import logging
import os
import socket
import sys

from fcest.models.sliding_windows import SlidingWindows
import pandas as pd

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    model_name = 'SW_cross_validated'

    data_set_name = sys.argv[1]    # 'd2', 'd3d', 'd{%d}s'
    data_split = sys.argv[2]       # 'all', 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    num_trials = int(experiment_data[-4:])

    # Allow for local and CPU cluster training.
    # When running on the Hivemind with SLURM, only one model is trained here.
    # TODO: this doesn't work with saving the optimal window length
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
            i_trials = range(num_trials)
            noise_types = cfg['noise-types']
            covs_types = cfg['all-covs-types']
    else:
        print('Running locally...')
        i_trials = range(num_trials)
        noise_types = cfg['noise-types']
        covs_types = cfg['all-covs-types']

    for noise_type in noise_types:

        optimal_window_length_df = pd.DataFrame()

        for covs_type in covs_types:

            optimal_window_length_array = []

            for i_trial in i_trials:

                print('\n----------')
                print(f'Trial      {i_trial+1:d} / {len(i_trials):d}')
                print('covs_type ', covs_type)
                print('noise_type', noise_type, '\n----------\n')

                data_file = os.path.join(
                    cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                    f'{covs_type:s}_covariance.csv'
                )
                if not os.path.exists(data_file):
                    logging.warning(f"Data file {data_file:s} not found.")

                    if covs_type == 'boxcar':
                        data_file = os.path.join(
                            cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                            'checkerboard_covariance.csv'
                        )
                        if not os.path.exists(data_file):
                            logging.warning(f"Data file {data_file:s} not found.")
                            continue
                    else:
                        continue

                x, y = load_data(
                    data_file,
                    verbose=False,
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
                        logging.error("Data split not recognized.")

                m = SlidingWindows(
                    x_train_locations=x_train,
                    y_train_locations=y_train,
                )
                optimal_window_length = m.compute_cross_validated_optimal_window_length()

                for metric in ['correlation', 'covariance']:
                    tvfc_estimates_savedir = os.path.join(
                        cfg['experiments-basedir'], noise_type, f'trial_{i_trial:03d}',
                        'TVFC_estimates', data_split, metric, model_name
                    )
                    m.save_tvfc_estimates(
                        optimal_window_length=optimal_window_length,
                        savedir=tvfc_estimates_savedir,
                        model_name=f'{covs_type:s}.csv',
                        connectivity_metric=metric,
                    )

                optimal_window_length_array.append(optimal_window_length)

            if len(optimal_window_length_array) > 0:
                optimal_window_length_df[covs_type] = optimal_window_length_array  # (n_trials, )

        optimal_window_length_filename = 'optimal_window_lengths.csv'
        optimal_window_length_savedir = os.path.join(
            cfg['git-results-basedir'], noise_type, data_split
        )
        if not os.path.exists(optimal_window_length_savedir):
            os.makedirs(optimal_window_length_savedir)
        optimal_window_length_df.to_csv(
            os.path.join(optimal_window_length_savedir, optimal_window_length_filename)
        )
        logging.info(
            f"Saved optimal window lengths '{optimal_window_length_filename:s}' in '{optimal_window_length_savedir:s}'."
        )
