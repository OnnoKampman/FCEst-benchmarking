import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import reorder_ica_components
from helpers.hcp import scan_id_to_scan_name


def _plot_brain_state_cluster_centroids(
        config_dict: dict, model_name: str, n_basis_states: int, data_dimensionality: str,
        connectivity_metric: str = 'correlation', figures_savedir: str = None
) -> None:
    """
    Plot brain states.
    TODO: add RSN names labels to brain states in plot?
    TODO: align brain states across scans automatically

    :param config_dict:
    :param model_name:
    :param n_basis_states:
    :param data_dimensionality:
    :param connectivity_metric:
    :param figures_savedir:
    """
    sns.set(style="whitegrid", font_scale=0.8)
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, axn = plt.subplots(
        nrows=len(config_dict['scan-ids']),
        ncols=n_basis_states,
        sharex=True,
        sharey=True,
    )
    cbar_ax = fig.add_axes([.91, .39, .03, .2])

    for scan_id in config_dict['scan-ids']:
        brain_states_savedir = os.path.join(
            config_dict['git-results-basedir'], 'brain_states',
            f'k{n_basis_states:02d}', f'scan_{scan_id:d}'
        )
        for brain_state in np.arange(n_basis_states):
            cluster_centroid_df = pd.read_csv(
                os.path.join(
                    brain_states_savedir, f'{connectivity_metric:s}_brain_state_{brain_state:d}_{model_name:s}.csv'
                ),
                index_col=0
            )  # (D, D)
            n_time_series = cluster_centroid_df.shape[0]  # D

            brain_state += 1  # make 1-indexed
            i_subplot = scan_id * n_basis_states + brain_state

            ax = plt.subplot(len(config_dict['scan-ids']), n_basis_states, i_subplot)

            if data_dimensionality == 'd15':
                cluster_centroid_array, new_rsn_names = reorder_ica_components(
                    original_matrix=cluster_centroid_df.values,
                    n_time_series=n_time_series,
                    config_dict=config_dict
                )
            else:
                cluster_centroid_array = cluster_centroid_df.values

            sns.heatmap(
                cluster_centroid_array,
                ax=ax,
                cmap='jet',  # 'viridis'
                cbar=i_subplot == 1,
                vmin=-0.75,
                vmax=0.75,
                cbar_ax=None if i_subplot != 1 else cbar_ax,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            # plt.xticks(fontsize=12)
            # plt.yticks(rotation=0, fontsize=12)
            # plt.xlabel("ICA component", fontsize=12)
            # plt.ylabel("ICA component", fontsize=12)

            if scan_id == 0:
                plt.title(f"Brain state {brain_state:d}")
            if brain_state == 1:
                scan_name = scan_id_to_scan_name(scan_id)
                plt.ylabel(f"Scan {scan_name:s}")

    fig.subplots_adjust(hspace=-0.1, wspace=-0.7)

    # fig.tight_layout(rect=[0, 0, .9, 1])
    fig.tight_layout()

    if figures_savedir is not None:
        figure_name = f'brain_states_{model_name:s}_k{n_basis_states:d}.pdf'
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_brain_states_cluster_centroids_joint():
    """
    Plot joint figure of various estimation methods of interest.
    TODO: could we quantify the similarity between the brain states across methods?
    """
    raise NotImplementedError


if __name__ == "__main__":

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    model_name = sys.argv[2]           # 'SVWP_joint', 'DCC_joint', 'SW_cross_validated', 'SW_30', 'SW_60'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_brain_states_list = cfg['n-brain-states-list']

    for n_brain_states in n_brain_states_list:
        _plot_brain_state_cluster_centroids(
            config_dict=cfg,
            model_name=model_name,
            n_basis_states=n_brain_states,
            data_dimensionality=data_dimensionality,
            figures_savedir=os.path.join(cfg['figures-basedir'], 'brain_states')
        )
