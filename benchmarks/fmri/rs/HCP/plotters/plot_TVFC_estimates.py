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
from helpers.plotters import plot_method_tvfc_estimates


def plot_hcp_tvfc_estimates(
    config_dict: dict,
    x_train_locations: np.array,
    y_train_locations: np.array,
    metric: str,
    data_split: str,
    scan_id: int,
    experiment_dimensionality: str,
    subject: int,
    random_edges: bool = False,
    figsize: tuple[float] = (6.3, 5.1),
    figures_savedir: str = None,
) -> None:
    """
    Plots estimated TVFC for a specified or random selection of edges.

    TODO: can we make lines less opaque so that orange and green are both visible?

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.
    x_train_locations : np.array
        Time points of the training data.
    y_train_locations : np.array
        Time series data.
    metric : str
        Connectivity metric.
    data_split : str
        Data split.
    scan_id : int
        Scan ID.
    experiment_dimensionality : str
    subject: int
    random_edges: bool
    figsize: tuple[float]
    figures_savedir: str
    """
    num_time_series = y_train_locations.shape[1]

    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    num_rows = 3
    num_columns = 2
    fig, ax = plt.subplots(
        nrows=num_rows,
        ncols=num_columns,
        # figsize=config_dict['plot-model-estimates-figsize'],
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if random_edges:
        interaction_pairs_indices = np.triu_indices(num_time_series, k=1)  # set k=0 to include variances
        interaction_pairs_indices = np.array(interaction_pairs_indices).T
        n_interactions = int(num_time_series * (num_time_series - 1) / 2)
        assert n_interactions == len(interaction_pairs_indices)
        random_interactions = np.random.choice(len(interaction_pairs_indices), num_rows*num_columns)
        interaction_pairs_indices = interaction_pairs_indices[random_interactions]
    else:
        # Pick some edges based on which ones show high imputation benchmark performance.
        edges_of_interest_indices = [
            (0, 2),   # V(L) - V(M)
            (1, 9),
            (4, 10),  # AUD - CBM
            (4, 11),  # AUD - SM
            (2, 3),   # V(M) - V(O)
            (12, 14),
        ]
        n_interactions = len(edges_of_interest_indices)
        assert n_interactions == num_rows * num_columns
        interaction_pairs_indices = edges_of_interest_indices

    for i_interaction, (i_time_series, j_time_series) in enumerate(interaction_pairs_indices):
        print(f'\nEdge {i_interaction+1:d} / {n_interactions:d}: time series {i_time_series:d} <-> {j_time_series:d}')

        splot = plt.subplot(num_rows, num_columns, i_interaction + 1)

        for tvfc_estimation_method in config_dict['plot-model-estimates-methods'][:-1]:  # remove sFC plot

            plot_method_tvfc_estimates(
                config_dict=config_dict,
                model_name=tvfc_estimation_method,
                i_time_series=i_time_series,
                j_time_series=j_time_series,
                x_train_locations=x_train_locations,
                y_train_locations=y_train_locations,
                data_split=data_split,
                scan_id=scan_id,
                experiment_dimensionality=experiment_dimensionality,
                subject=subject,
                metric=metric,
            )

        # Plot uncorrelated reference line.
        # plt.plot(
        #     x_train_locations, np.zeros_like(y_train_locations),
        #     linestyle='dashed', color='black', linewidth=1.2
        # )

        bbox_to_anchor = (1.02, 1.0)  # used to put legend outside of plot
        if i_interaction == (num_columns - 1):
            plt.legend(
                bbox_to_anchor=bbox_to_anchor,
                frameon=True,
                title='TVFC\nestimator',
                alignment='left',
            )

        # plt.gca().get_xaxis().set_visible(False)
        # plt.gca().get_yaxis().set_visible(False)

        plt.xlim([0.0, 14.6])
        plt.ylim([-1.0, 1.0])
        if data_dimensionality == 'd15':
            plt.title(f"{rsn_names[i_time_series]:s} - {rsn_names[j_time_series]:s}")
        else:
            plt.title(f"{i_time_series:d} - {j_time_series:d}")
        # splot.axis('off')

    # plt.gca().get_xaxis().set_visible(True)

    ax[2, 0].set_xlabel('time [minutes]')
    ax[2, 1].set_xlabel('time [minutes]')

    ax[0, 0].set_ylabel('TVFC estimates')
    ax[1, 0].set_ylabel('TVFC estimates')
    ax[2, 0].set_ylabel('TVFC estimates')

    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.1, wspace=0.2)  # only for when the axis are turned off

    if figures_savedir is not None:
        if random_edges:
            figure_name = f'{metric:s}_estimates_random_edges.pdf'
        else:
            figure_name = f'{metric:s}_estimates_edges_of_interest.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'
    metric = 'correlation'
    subjects = [
        100206
    ]
    scan_ids = [
        0,
        1,
    ]

    data_dimensionality = sys.argv[1]  # 'd15' or 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    rsn_names_dict = cfg['rsn-id-to-functional-region-map']
    print(rsn_names_dict)
    ica_to_rsn_map = cfg['ica-id-to-rsn-id-algo-map']
    print(ica_to_rsn_map)
    if ica_to_rsn_map is not None:
        rsn_names = [rsn_names_dict[ica_to_rsn_map[i]] for i in range(15)]
        print(rsn_names)
    else:
        rsn_names = list(rsn_names_dict.values())
        print(rsn_names)

    for subject in subjects:
        logging.info(f'SUBJECT {subject:d}')
        data_file = os.path.join(cfg['data-dir'], f'{subject:d}.txt')
        for scan_id in scan_ids:
            print(f'\nSCAN ID {scan_id:d}\n')
            figures_savedir = os.path.join(
                cfg['figures-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, str(subject)
            )
            if not os.path.exists(figures_savedir):
                os.makedirs(figures_savedir)

            x, y = load_human_connectome_project_data(
                data_file, scan_id=scan_id, verbose=False
            )  # (N, 1), (N, D)

            if data_split == 'LEOO':
                x_train, _ = leave_every_other_out_split(x)
                y_train, _ = leave_every_other_out_split(y)
            else:
                x_train = x
                y_train = y

            num_time_steps = x_train.shape[0]
            xx = convert_to_minutes(
                x_train,
                repetition_time=cfg['repetition-time'],
                data_length=num_time_steps
            )
            plot_hcp_tvfc_estimates(
                config_dict=cfg,
                x_train_locations=xx,
                y_train_locations=y_train,
                metric=metric,
                data_split=data_split,
                scan_id=scan_id,
                experiment_dimensionality=experiment_dimensionality,
                subject=subject,
                random_edges=False,
                figures_savedir=figures_savedir,
            )
            plot_hcp_tvfc_estimates(
                config_dict=cfg,
                x_train_locations=xx,
                y_train_locations=y_train,
                metric=metric,
                data_split=data_split,
                scan_id=scan_id,
                experiment_dimensionality=experiment_dimensionality,
                subject=subject,
                random_edges=True,
                figures_savedir=figures_savedir,
            )
