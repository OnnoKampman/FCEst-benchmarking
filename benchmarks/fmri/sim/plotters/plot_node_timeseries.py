import logging
import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split
from helpers.plotters import plot_node_timeseries


if __name__ == "__main__":

    data_split = 'all'

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    experiment_data = sys.argv[2]  # e.g. 'N0200_T0100'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    num_trials = int(experiment_data[-4:])
    assert os.path.exists(os.path.join(cfg['data-dir']))

    noise_types = cfg['noise-types']
    covs_types = cfg['all-covs-types']

    for noise_type in noise_types:
        for covs_type in covs_types:
            for i_trial in range(num_trials):
                print(f"\n> TRIAL {i_trial+1:d} / {num_trials:d}\n")

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
                        if not os.path.exists(data_file):
                            logging.warning(f"File '{data_file:s}' not found.")
                            continue

                figures_savedir = os.path.join(
                    cfg['figures-basedir'], noise_type, data_split, 'time_series',
                    f'trial_{i_trial:03d}'
                )
                if not os.path.exists(figures_savedir):
                    os.makedirs(figures_savedir)

                x, y = load_data(
                    data_file,
                    verbose=False,
                )  # (N, 1), (N, D)
                match data_split:
                    case "LEOO":
                        x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), _
                        y_train, _ = leave_every_other_out_split(y)  # (N/2, D), _
                    case "all":
                        x_train = x
                        y_train = y
                    case _:
                        logging.error("Data split not recognized.")

                n_time_steps = x_train.shape[0]
                plot_node_timeseries(
                    config_dict=cfg,
                    x_plot=x_train,
                    y_locations=y_train,
                    figures_savedir=figures_savedir
                )
