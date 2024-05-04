import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.hcp import load_human_connectome_project_data
from helpers.plotters import convert_to_minutes

sns.set(style="whitegrid", font_scale=1.6)
font = {
    'family': 'Times New Roman',
#     'weight': 'normal',
#     'size': 14
}
plt.rc('font', **font)


def _plot_time_series(
        config_dict: dict, x_plot: np.array, y_locations: np.array
) -> None:
    """
    Plots BOLD time series per brain region.
    """
    figure_name_time_series = 'time_series.pdf'

    n_time_series = y_locations.shape[1]

    plt.figure(figsize=config_dict['plot-time-series-figsize'])
    for i_time_series in range(n_time_series):
        plt.subplot(n_time_series, 1, i_time_series+1)
        plt.plot(
            x_plot, y_locations[:, i_time_series], 'x-',
            markersize=0, label=f'TS_{(i_time_series+1):d}'
        )
        plt.xlim(config_dict['plot-time-series-xlim'])
        plt.ylim(config_dict['plot-time-series-ylim'])
        plt.yticks([0], [i_time_series+1])
        plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_xaxis().set_visible(True)
    plt.xlabel('time [minutes]')
    # plt.tight_layout()
    plt.savefig(
        os.path.join(figures_savedir, figure_name_time_series),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name_time_series:s}' in '{figures_savedir:s}'.")
    plt.close()


if __name__ == "__main__":

    data_split = 'all'
    subjects = [
        100206
    ]
    scan_ids = [
        0,
        1
    ]

    data_dimensionality = sys.argv[1]  # 'd15' or 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    for i_subject, subject in enumerate(subjects):
        print(f"\nSUBJECT {i_subject+1:d}: {subject:d}\n")
        data_file = os.path.join(cfg['data-dir'], f'{subject:d}.txt')
        for scan_id in scan_ids:
            print(f'\nSCAN ID {scan_id:d}\n')
            figures_savedir = os.path.join(cfg['figures-basedir'], f'scan_{scan_id:d}', data_split, str(subject))
            if not os.path.exists(figures_savedir):
                os.makedirs(figures_savedir)

            x, y = load_human_connectome_project_data(data_file, scan_id=scan_id, verbose=False)  # (N, 1), (N, D)

            match data_split:
                case 'LEOO':
                    x_train, _ = leave_every_other_out_split(x)
                    y_train, _ = leave_every_other_out_split(y)
                case _:
                    x_train = x
                    y_train = y

            n_time_steps = x_train.shape[0]
            xx = convert_to_minutes(
                x_train,
                repetition_time=cfg['repetition-time'],
                data_length=n_time_steps
            )
            _plot_time_series(
                config_dict=cfg,
                x_plot=xx,
                y_locations=y_train
            )
